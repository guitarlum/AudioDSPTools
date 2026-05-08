//
//  Reverb.cpp
//  VoLum - Hall (FDN), Plate (Dattorro), Oktaverb, TremVerb
//

#include "Reverb.h"
#include <algorithm>
#include <cmath>

namespace dsp
{
namespace effect
{

static constexpr double kPI = 3.14159265358979323846;
static constexpr double kTwoPI = 2.0 * kPI;

static size_t ScaleDelay(double ms, double sr) { return std::max<size_t>(1, static_cast<size_t>(ms * sr / 1000.0)); }
static size_t ScaleFromRef(size_t refLen, double refSR, double sr) { return std::max<size_t>(1, static_cast<size_t>(refLen * sr / refSR)); }
static double Hann(double phase) { return 0.5 - 0.5 * std::cos(kTwoPI * phase); }

static double LPTick(double& state, double in, double coef) { state += coef * (in - state); return state; }

static double AllpassTick(std::vector<double>& buf, size_t& idx, size_t len, double input, double coef)
{
  if (len == 0 || buf.empty()) return input;
  size_t bufSz = buf.size();
  size_t readIdx = (idx + bufSz - len) % bufSz;
  double delayed = buf[readIdx];
  double v = input + coef * delayed;
  buf[idx] = v;
  idx = (idx + 1) % bufSz;
  return delayed - coef * v;
}

namespace
{

// Hall sub-mode scale factors. 0=Studio (smaller/brighter), 1=Concert (default),
// 2=Cathedral (longer/darker, with chorused tail).
struct HallSubModeChar
{
  double lengthScale;
  double cutoffOctaveOffset; // added to log2(cutoff)
  double modDepthScale;
  bool   chorusedTail;
};

HallSubModeChar GetHallSubMode(int sub)
{
  switch (sub)
  {
    case 0: return {0.50, +1.5, 0.5, false}; // Studio
    case 2: return {1.70, -0.5, 1.5, true};  // Cathedral
    case 1:
    default: return {1.00, 0.0, 1.0, false}; // Concert
  }
}

// Plate sub-mode: Steel (default), Brass (sharp/bright), Copper (chamber-like).
struct PlateSubModeChar
{
  double decayApCoef;       // tank AP coefficient
  double tankLengthScale;   // delay-line scaling
  double targetCutoffHz;    // damping LP target
  double modDepthScale;
  int    erTapCount;        // number of early-reflection taps
};

PlateSubModeChar GetPlateSubMode(int sub)
{
  switch (sub)
  {
    case 1: return {0.45, 0.80, 8000.0, 0.8, 6}; // Brass
    case 2: return {0.70, 1.30, 4000.0, 1.5, 8}; // Copper
    case 0:
    default: return {0.55, 1.00, 6000.0, 1.0, 4}; // Steel
  }
}

// Oktaverb sub-mode:
//   0 = Oct (octave up only)
//   1 = Oct + Fifth
//   2 = Oct + SubOct (preserves the previous sub-octave behaviour)
struct OktaverbSubModeChar
{
  double octGain;
  double fifthGain;
  double subOctGain;
  double pitchedPreDelayMs;
};

OktaverbSubModeChar GetOktaverbSubMode(int sub)
{
  switch (sub)
  {
    case 1: return {1.0, 0.7, 0.0, 80.0};
    case 2: return {0.8, 0.0, 0.6, 50.0};
    case 0:
    default: return {1.0, 0.0, 0.0, 60.0};
  }
}

constexpr double kFifthRatio = 1.4983;
constexpr double kOctaveUpRatio = 2.0;
constexpr double kOctaveDownRatio = 0.5;

} // namespace

Reverb::Reverb()
{
  mHallDelays.resize(kHallLines);
  mHallIndices.resize(kHallLines, 0);
  mHallLengths.resize(kHallLines, 0);
  mHallLPState.resize(kHallLines, 0.0);
  mHallChorusPhase.resize(kHallLines, 0.0);

  mInputAPBuf.resize(kInputAPs);
  mInputAPIdx.resize(kInputAPs, 0);
  mInputAPLen.resize(kInputAPs, 0);

  mPitchBufs.resize(kNumPitchVoices);
  mPitchWriteIdx.resize(kNumPitchVoices);
  mPitchPhase.resize(kNumPitchVoices);
  for (int v = 0; v < kNumPitchVoices; v++)
  {
    mPitchBufs[v].resize(kHallLines);
    mPitchWriteIdx[v].assign(kHallLines, 0);
    mPitchPhase[v].assign(kHallLines, 0.0);
  }

  mPitchedPreBuf.resize(kHallLines);
  mPitchedPreIdx.assign(kHallLines, 0);
  mPitchedDetunePhase.assign(kHallLines, 0.0);
}

void Reverb::Prepare(const size_t numChannels, const size_t numFrames, double sampleRate)
{
  const bool srChanged = (mSampleRate != sampleRate);
  mSampleRate = sampleRate;
  _PrepareBuffers(numChannels, numFrames);

  if (srChanged)
  {
    mHallAllocated = false;
    mPlateAllocated = false;
    mOktaverbAllocated = false;
    _AllocatePreDelay();
  }
  else if (mPreDelayBuf.empty())
  {
    _AllocatePreDelay();
  }

  if (!mHallAllocated) _AllocateHall();
  if (!mPlateAllocated) _AllocatePlate();
  if (!mOktaverbAllocated) _AllocateOktaverb();
}

void Reverb::SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode,
                       double sampleRate)
{
  // Backward-compat 7-arg API: subMode defaults to 1 (the legacy "middle" / Concert / Steel /
  // Oct selection chosen per-mode below) and tremRate defaults to 4 Hz.
  // Map legacy subMode=1 to the original behaviour for each mode by using its old default.
  int legacyDefaultSub = 1;
  if (mode == kModePlate) legacyDefaultSub = 0;     // Steel was the implicit default
  if (mode == kModeOktaverb) legacyDefaultSub = 0;  // Oct (sub-octave behaviour preserved via mode 2)
  // For Hall the new "Concert" sub-mode (1) matches the existing length scaling.
  SetParams(mix, decay, tone, preDelayMs, shimmer, mode, sampleRate, legacyDefaultSub, 4.0);
}

void Reverb::SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode,
                       double sampleRate, int subMode, double tremRateHz)
{
  const bool srChanged = (mSampleRate != sampleRate);
  // Pre-delay range expanded to 200 ms in v0.9.0; clamp accordingly.
  const double clampedPreDelayMs = std::clamp(preDelayMs, 0.0, 200.0);
  mSampleRate = sampleRate;
  mMix = std::clamp(mix, 0.0, 1.0);
  mDecay = std::clamp(decay, 0.1, 10.0);
  mTone = std::clamp(tone, 0.0, 10.0);
  _SetPreDelayLength(clampedPreDelayMs);
  mShimmer = std::clamp(shimmer, 0.0, 1.0);
  mMode = std::clamp(mode, 0, kNumModes - 1);
  mSubMode = std::clamp(subMode, 0, 2);
  mTremRateHz = std::clamp(tremRateHz, 0.5, 10.0);

  if (srChanged)
  {
    mHallAllocated = false;
    mPlateAllocated = false;
    mOktaverbAllocated = false;
    _AllocatePreDelay();
  }
  else if (mPreDelayBuf.empty())
  {
    _AllocatePreDelay();
  }

  if (mMode == kModeHall && !mHallAllocated)
    _AllocateHall();
  if (mMode == kModePlate && !mPlateAllocated)
    _AllocatePlate();
  if (mMode == kModeOktaverb)
  {
    if (!mHallAllocated) _AllocateHall();
    if (!mOktaverbAllocated) _AllocateOktaverb();
  }
  if (mMode == kModeTremVerb && !mPlateAllocated)
    _AllocatePlate();
}

void Reverb::_AllocatePreDelay()
{
  // Maximum pre-delay extended to 200 ms (was 80 ms) to support the new TremVerb/Oktaverb
  // pre-delay range.
  const size_t maxPreDelayLen = ScaleDelay(200.0, mSampleRate);
  mPreDelayLen = static_cast<size_t>(mPreDelayMs * mSampleRate / 1000.0);
  mPreDelayBuf.assign(maxPreDelayLen + 4, 0.0);
  mPreDelayIdx = 0;
}

void Reverb::_SetPreDelayLength(double preDelayMs)
{
  mPreDelayMs = preDelayMs;
  mPreDelayLen = static_cast<size_t>(mPreDelayMs * mSampleRate / 1000.0);
  if (!mPreDelayBuf.empty() && mPreDelayLen >= mPreDelayBuf.size())
    mPreDelayLen = mPreDelayBuf.size() - 1;
}

void Reverb::_AllocateHall()
{
  // Base prime-ish lengths (Concert default).
  const double baseLengthsMs[kHallLines] = {31.0, 37.0, 43.0, 53.0, 61.0, 71.0, 79.0, 89.0};
  // Headroom for sub-mode scaling: Cathedral scales by 1.7, so we size buffers for the worst
  // case to avoid reallocation when the user toggles sub-modes.
  const double headroom = 1.8;
  for (int i = 0; i < kHallLines; i++)
  {
    mHallLengths[i] = ScaleDelay(baseLengthsMs[i], mSampleRate);
    const size_t bufSz = static_cast<size_t>(mHallLengths[i] * headroom) + 200;
    mHallDelays[i].assign(bufSz, 0.0);
    mHallIndices[i] = 0;
    mHallLPState[i] = 0.0;
    mHallChorusPhase[i] = 0.0;
  }
  mHallLfoPhase = 0.0;
  mHallAllocated = true;
}

void Reverb::_AllocateOktaverb()
{
  const size_t grainLen = std::max<size_t>(64, ScaleDelay(60.0, mSampleRate));
  for (int v = 0; v < kNumPitchVoices; v++)
  {
    for (int i = 0; i < kHallLines; i++)
    {
      mPitchBufs[v][i].assign(grainLen * 2 + 4, 0.0);
      mPitchWriteIdx[v][i] = 0;
      mPitchPhase[v][i] = 0.0;
    }
  }
  // Pitched pre-delay buffer per line (sized for max 120 ms).
  const size_t maxPitchedPre = ScaleDelay(150.0, mSampleRate);
  for (int i = 0; i < kHallLines; i++)
  {
    mPitchedPreBuf[i].assign(maxPitchedPre + 4, 0.0);
    mPitchedPreIdx[i] = 0;
    mPitchedDetunePhase[i] = static_cast<double>(i) / kHallLines; // staggered phase
  }
  mOktaverbAllocated = true;
}

void Reverb::_AllocatePlate()
{
  const double refSR = 29761.0;

  const size_t inputAPRef[kInputAPs] = {142, 107, 379, 277};
  for (int i = 0; i < kInputAPs; i++)
  {
    mInputAPLen[i] = ScaleFromRef(inputAPRef[i], refSR, mSampleRate);
    mInputAPBuf[i].assign(mInputAPLen[i] + 16, 0.0);
    mInputAPIdx[i] = 0;
  }

  const size_t tankAPRef[2] = {672, 908};
  const size_t tankDelRef[2] = {4453, 4217};

  // Plate buffers are sized with headroom for sub-mode scaling (Copper is 1.3x).
  const double headroom = 1.5;
  for (int h = 0; h < 2; h++)
  {
    mTank[h].apLen = ScaleFromRef(tankAPRef[h], refSR, mSampleRate);
    mTank[h].apBuf.assign(static_cast<size_t>(mTank[h].apLen * headroom) + 32, 0.0);
    mTank[h].apIdx = 0;

    mTank[h].delLen = ScaleFromRef(tankDelRef[h], refSR, mSampleRate);
    mTank[h].delBuf.assign(static_cast<size_t>(mTank[h].delLen * headroom) + 32, 0.0);
    mTank[h].delIdx = 0;

    mTank[h].lpState = 0.0;
    mTank[h].lastOut = 0.0;
  }
  mInputLPState = 0.0;
  mPlateLfoPhase = 0.0;

  // Early-reflection taps. Maximum 8 taps; configured per sub-mode.
  mErTapMs = {5.0, 9.0, 13.0, 19.0, 23.0, 27.0, 31.0, 37.0};
  mErTapGain = {0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15};

  mPlateAllocated = true;
}

void Reverb::Reset()
{
  for (int i = 0; i < kHallLines; i++)
  {
    std::fill(mHallDelays[i].begin(), mHallDelays[i].end(), 0.0);
    mHallIndices[i] = 0;
    mHallLPState[i] = 0.0;
    mHallChorusPhase[i] = 0.0;
  }
  for (int i = 0; i < kInputAPs; i++)
  {
    std::fill(mInputAPBuf[i].begin(), mInputAPBuf[i].end(), 0.0);
    mInputAPIdx[i] = 0;
  }
  for (int h = 0; h < 2; h++)
  {
    std::fill(mTank[h].apBuf.begin(), mTank[h].apBuf.end(), 0.0);
    mTank[h].apIdx = 0;
    std::fill(mTank[h].delBuf.begin(), mTank[h].delBuf.end(), 0.0);
    mTank[h].delIdx = 0;
    mTank[h].lpState = 0.0;
    mTank[h].lastOut = 0.0;
  }
  std::fill(mPreDelayBuf.begin(), mPreDelayBuf.end(), 0.0);
  mPreDelayIdx = 0;
  for (int v = 0; v < kNumPitchVoices; v++)
    for (int i = 0; i < kHallLines; i++)
    {
      std::fill(mPitchBufs[v][i].begin(), mPitchBufs[v][i].end(), 0.0);
      mPitchWriteIdx[v][i] = 0;
      mPitchPhase[v][i] = 0.0;
    }
  for (int i = 0; i < kHallLines; i++)
  {
    std::fill(mPitchedPreBuf[i].begin(), mPitchedPreBuf[i].end(), 0.0);
    mPitchedPreIdx[i] = 0;
    mPitchedDetunePhase[i] = static_cast<double>(i) / kHallLines;
  }
  mInputLPState = 0.0;
  mHallLfoPhase = 0.0;
  mPlateLfoPhase = 0.0;
  mTremPhase = 0.0;
}

void Reverb::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  this->DSP::_PrepareBuffers(numChannels, numFrames);
}

DSP_SAMPLE** Reverb::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

  switch (mMode)
  {
    case kModeOktaverb: _ProcessOktaverb(inputs, numChannels, numFrames); break;
    case kModePlate: _ProcessPlate(inputs, numChannels, numFrames); break;
    case kModeTremVerb: _ProcessTremVerb(inputs, numChannels, numFrames); break;
    case kModeHall:
    default: _ProcessHall(inputs, numChannels, numFrames); break;
  }

  return _GetPointers();
}

double Reverb::_ReadWritePreDelay(double input)
{
  if (mPreDelayBuf.empty())
    return input;
  if (mPreDelayLen == 0)
    return input;

  const size_t readIdx = (mPreDelayIdx + mPreDelayBuf.size() - mPreDelayLen) % mPreDelayBuf.size();
  const double out = mPreDelayBuf[readIdx];
  mPreDelayBuf[mPreDelayIdx] = input;
  mPreDelayIdx = (mPreDelayIdx + 1) % mPreDelayBuf.size();
  return out;
}

double Reverb::_PitchShiftTick(int voice, int line, double input, double ratio)
{
  auto& buf = mPitchBufs[voice][line];
  if (buf.empty())
    return input;

  size_t& writeIdx = mPitchWriteIdx[voice][line];
  double& phase = mPitchPhase[voice][line];
  const size_t grainLen = std::max<size_t>(2, (buf.size() - 4) / 2);

  buf[writeIdx] = input;

  auto readFrac = [&](double grainPhase) {
    const double delay = 1.0 + grainPhase * static_cast<double>(grainLen);
    double readPos = static_cast<double>(writeIdx) - delay;
    while (readPos < 0.0)
      readPos += static_cast<double>(buf.size());
    const size_t i0 = static_cast<size_t>(readPos) % buf.size();
    const size_t i1 = (i0 + 1) % buf.size();
    const double frac = readPos - std::floor(readPos);
    return buf[i0] * (1.0 - frac) + buf[i1] * frac;
  };

  const double phaseB = phase + 0.5 >= 1.0 ? phase - 0.5 : phase + 0.5;
  const double a = readFrac(phase);
  const double b = readFrac(phaseB);
  const double wA = Hann(phase);
  const double wB = Hann(phaseB);
  const double out = (a * wA + b * wB) / std::max(0.000001, wA + wB);

  writeIdx = (writeIdx + 1) % buf.size();
  // For pitch ratio R, advance by (1 - 1/R) per sample; for R=2 (octave up) this is 0.5,
  // matching the previous behaviour's grain advance.
  const double advance = std::max(0.0, 1.0 - 1.0 / ratio) / static_cast<double>(grainLen);
  phase += advance;
  while (phase >= 1.0)
    phase -= 1.0;

  return out;
}

double Reverb::_ToneToCutoff(double minHz, double midHz, double maxHz) const
{
  // Tone is 0..10. Tone=5 ("noon") maps to midHz; left half (0..5) maps log to [minHz, midHz];
  // right half (5..10) maps log to [midHz, maxHz]. The dark side is compressed by using a
  // perceptually log-linear mapping so the bottom portion of the range is much less extreme
  // than a linear mapping was.
  const double t = std::clamp(mTone / 10.0, 0.0, 1.0);
  if (t <= 0.5)
  {
    const double f = t / 0.5; // 0..1
    return std::pow(midHz, f) * std::pow(minHz, 1.0 - f);
  }
  const double f = (t - 0.5) / 0.5; // 0..1
  return std::pow(maxHz, f) * std::pow(midHz, 1.0 - f);
}

// ─── Hall ─────────────────────────────────────────────────────────

void Reverb::_ProcessHall(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const HallSubModeChar sm = GetHallSubMode(mSubMode);

  // Compute effective per-line lengths from base lengths × sub-mode scale.
  size_t effLengths[kHallLines];
  for (int i = 0; i < kHallLines; i++)
  {
    effLengths[i] = static_cast<size_t>(mHallLengths[i] * sm.lengthScale);
    effLengths[i] = std::min(effLengths[i], mHallDelays[i].size() - 4);
    if (effLengths[i] < 1)
      effLengths[i] = 1;
  }

  const double avgDelay = 58.0 / 1000.0 * sm.lengthScale;
  double loopGain = std::pow(10.0, -3.0 * avgDelay / mDecay);
  loopGain = std::min(loopGain, 0.998);

  // Tone curve: dark side compressed; sub-mode shifts cutoff up/down.
  const double cutoffBase = _ToneToCutoff(1500.0, 5000.0, 10000.0);
  const double cutoffHz = std::clamp(cutoffBase * std::pow(2.0, sm.cutoffOctaveOffset), 500.0, 18000.0);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double lpCoef = dt / (rc + dt);

  const double H = 0.35355339;
  static const double mixMat[8][8] = {
    {H, H, H, H, H, H, H, H},
    {H, -H, H, -H, H, -H, H, -H},
    {H, H, -H, -H, H, H, -H, -H},
    {H, -H, -H, H, H, -H, -H, H},
    {H, H, H, H, -H, -H, -H, -H},
    {H, -H, H, -H, -H, H, -H, H},
    {H, H, -H, -H, -H, -H, H, H},
    {H, -H, -H, H, -H, H, H, -H}};

  const double lfoRate = 0.6 * 2.0 * kPI / mSampleRate;
  const double lfoDepth = 10.0 * sm.modDepthScale;

  // Cathedral: per-line slow detune on read tap (~0.2 Hz, 6 cents -> small frame fraction).
  const double cathRate = 0.2 / mSampleRate;
  const double cathDepth = sm.chorusedTail ? 4.0 : 0.0;

  for (size_t s = 0; s < numFrames; s++)
  {
    mHallLfoPhase += lfoRate;
    if (mHallLfoPhase > 2.0 * kPI) mHallLfoPhase -= 2.0 * kPI;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;
    in = _ReadWritePreDelay(in);

    double readVals[8];
    for (int i = 0; i < kHallLines; i++)
    {
      const double mod = std::sin(mHallLfoPhase + i * kPI * 0.25) * lfoDepth;
      double cathOffset = 0.0;
      if (sm.chorusedTail)
      {
        mHallChorusPhase[i] += cathRate;
        if (mHallChorusPhase[i] >= 1.0) mHallChorusPhase[i] -= 1.0;
        cathOffset = std::sin(2.0 * kPI * mHallChorusPhase[i]) * cathDepth;
      }

      double readPos = static_cast<double>(mHallIndices[i]) - static_cast<double>(effLengths[i]) - mod - cathOffset;
      const size_t bufSz = mHallDelays[i].size();
      while (readPos < 0.0) readPos += bufSz;
      const size_t i0 = static_cast<size_t>(readPos) % bufSz;
      const size_t i1 = (i0 + 1) % bufSz;
      const double frac = readPos - std::floor(readPos);
      double val = mHallDelays[i][i0] * (1.0 - frac) + mHallDelays[i][i1] * frac;

      LPTick(mHallLPState[i], val, lpCoef);
      readVals[i] = mHallLPState[i] * loopGain;
    }

    for (int i = 0; i < kHallLines; i++)
    {
      double fb = 0.0;
      for (int j = 0; j < kHallLines; j++) fb += mixMat[i][j] * readVals[j];
      double write = in * 0.5 + fb;
      if (!std::isfinite(write)) { write = 0.0; Reset(); }
      mHallDelays[i][mHallIndices[i]] = std::clamp(write, -3.0, 3.0);
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    const double outL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5;
    const double outR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── Oktaverb ──────────────────────────────────────────────────────

void Reverb::_ProcessOktaverb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const OktaverbSubModeChar sm = GetOktaverbSubMode(mSubMode);

  const double avgDelay = 58.0 / 1000.0;
  double loopGain = std::pow(10.0, -3.0 * avgDelay / mDecay);
  loopGain = std::min(loopGain, 0.998);

  const double cutoffHz = _ToneToCutoff(1500.0, 5000.0, 10000.0);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double lpCoef = dt / (rc + dt);

  const double H = 0.35355339;
  static const double mixMat[8][8] = {
    {H, H, H, H, H, H, H, H},
    {H, -H, H, -H, H, -H, H, -H},
    {H, H, -H, -H, H, H, -H, -H},
    {H, -H, -H, H, H, -H, -H, H},
    {H, H, H, H, -H, -H, -H, -H},
    {H, -H, H, -H, -H, H, -H, H},
    {H, H, -H, -H, -H, -H, H, H},
    {H, -H, -H, H, -H, H, H, -H}};

  const double lfoRate = 0.6 * 2.0 * kPI / mSampleRate;
  const double lfoDepth = 10.0;

  // Retuned shimmer curve: shimmer^1.5 makes 0.3 perceptually meaningful while still
  // letting cranked settings push hard.
  const double shimmerCurved = std::pow(mShimmer, 1.5);
  const double shimmerOutput = shimmerCurved * 0.95;

  // Pitched pre-delay frames.
  const size_t pitchedPreDelayFrames = ScaleDelay(sm.pitchedPreDelayMs, mSampleRate);

  // Per-line detune motion (3-6 cents at 0.3-0.5 Hz).
  const double detuneRateHz = 0.4;
  const double detuneCentDepth = 4.5;
  const double centsToFrameFrac = (detuneCentDepth / 1200.0); // tiny per-sample shift

  for (size_t s = 0; s < numFrames; s++)
  {
    mHallLfoPhase += lfoRate;
    if (mHallLfoPhase > 2.0 * kPI) mHallLfoPhase -= 2.0 * kPI;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;
    in = _ReadWritePreDelay(in);

    double readVals[8];
    double pitchedVals[8];
    for (int i = 0; i < kHallLines; i++)
    {
      const double mod = std::sin(mHallLfoPhase + i * kPI * 0.25) * lfoDepth;
      double readPos = static_cast<double>(mHallIndices[i]) - static_cast<double>(mHallLengths[i]) - mod;
      const size_t bufSz = mHallDelays[i].size();
      while (readPos < 0.0) readPos += bufSz;
      const size_t i0 = static_cast<size_t>(readPos) % bufSz;
      const size_t i1 = (i0 + 1) % bufSz;
      const double frac = readPos - std::floor(readPos);
      const double val = mHallDelays[i][i0] * (1.0 - frac) + mHallDelays[i][i1] * frac;

      LPTick(mHallLPState[i], val, lpCoef);
      const double feedback = mHallLPState[i] * loopGain;
      readVals[i] = feedback;

      // Pitched pre-delay: write the FDN read-tap into a per-line buffer; read it back N ms
      // later for the pitch shifter input. This adds the "pitched bloom arrives slightly
      // after the dry reverb onset" character.
      auto& preBuf = mPitchedPreBuf[i];
      const size_t preSz = preBuf.size();
      const size_t preLen = std::min(pitchedPreDelayFrames, preSz - 2);
      const size_t preReadIdx = (mPitchedPreIdx[i] + preSz - preLen) % preSz;
      const double preDelayedFeedback = preBuf[preReadIdx];
      preBuf[mPitchedPreIdx[i]] = feedback;
      mPitchedPreIdx[i] = (mPitchedPreIdx[i] + 1) % preSz;

      // Per-line detune: a tiny LFO on the input to the pitch shifter so the pitched bloom
      // moves over time.
      mPitchedDetunePhase[i] += detuneRateHz / mSampleRate;
      if (mPitchedDetunePhase[i] >= 1.0) mPitchedDetunePhase[i] -= 1.0;
      const double detune = std::sin(2.0 * kPI * mPitchedDetunePhase[i]) * centsToFrameFrac;

      // Compute pitched components.
      double pitched = 0.0;
      const double pitchInput = preDelayedFeedback * (1.0 + detune);
      if (sm.octGain > 0.0)
        pitched += _PitchShiftTick(kVoiceOctUp, i, pitchInput, kOctaveUpRatio) * sm.octGain;
      if (sm.fifthGain > 0.0)
        pitched += _PitchShiftTick(kVoiceFifthUp, i, pitchInput, kFifthRatio) * sm.fifthGain;
      if (sm.subOctGain > 0.0)
        pitched += _PitchShiftTick(kVoiceSubOct, i, pitchInput, kOctaveDownRatio) * sm.subOctGain;
      pitchedVals[i] = pitched;
    }

    for (int i = 0; i < kHallLines; i++)
    {
      double fb = 0.0;
      for (int j = 0; j < kHallLines; j++)
        fb += mixMat[i][j] * readVals[j];
      double write = in * 0.5 + fb;
      if (!std::isfinite(write)) { write = 0.0; Reset(); }
      mHallDelays[i][mHallIndices[i]] = std::clamp(write, -2.0, 2.0);
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    const double hallOutL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5;
    const double hallOutR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5;
    const double pitchOutL = (pitchedVals[0] + pitchedVals[2] - pitchedVals[4] + pitchedVals[6]) * 0.5;
    const double pitchOutR = (pitchedVals[1] - pitchedVals[3] + pitchedVals[5] + pitchedVals[7]) * 0.5;
    const double outL = hallOutL + pitchOutL * shimmerOutput;
    const double outR = hallOutR + pitchOutR * shimmerOutput;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── Plate ─────────────────────────────────────────────────────────

void Reverb::_ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const PlateSubModeChar sm = GetPlateSubMode(mSubMode);

  const double inDiff1 = 0.75;
  const double inDiff2 = 0.625;
  const double inDiffCoefs[kInputAPs] = {inDiff1, inDiff1, inDiff2, inDiff2};

  // Tank delay scale via sub-mode.
  size_t tankDelLen[2];
  for (int h = 0; h < 2; h++)
  {
    tankDelLen[h] = static_cast<size_t>(mTank[h].delLen * sm.tankLengthScale);
    tankDelLen[h] = std::min(tankDelLen[h], mTank[h].delBuf.size() - 4);
    if (tankDelLen[h] < 1) tankDelLen[h] = 1;
  }
  size_t tankApLen[2];
  for (int h = 0; h < 2; h++)
  {
    tankApLen[h] = static_cast<size_t>(mTank[h].apLen * sm.tankLengthScale);
    tankApLen[h] = std::min(tankApLen[h], mTank[h].apBuf.size() - 16);
    if (tankApLen[h] < 1) tankApLen[h] = 1;
  }

  double decayGain = std::clamp(1.0 - (1.0 / (mDecay + 0.1)), 0.1, 0.97);

  // Tone curve: brighter default than before.
  const double cutoffBase = _ToneToCutoff(1800.0, 6000.0, 11000.0);
  const double cutoffHz = std::min(cutoffBase, sm.targetCutoffHz);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double dampCoef = dt / (rc + dt);

  const double bwCoef = 0.9995;

  // Faster modulation for v0.9.0 (was ~1 Hz, now 1.5 Hz default scaled by sub-mode).
  const double modRate = 1.5 * sm.modDepthScale * 2.0 * kPI / mSampleRate;
  const double modDepth = 16.0 * sm.modDepthScale * mSampleRate / 29761.0;

  // Early-reflection taps (fixed count from sub-mode).
  const int erCount = std::min<int>(sm.erTapCount, static_cast<int>(mErTapMs.size()));

  for (size_t s = 0; s < numFrames; s++)
  {
    mPlateLfoPhase += modRate;
    if (mPlateLfoPhase > 2.0 * kPI) mPlateLfoPhase -= 2.0 * kPI;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;
    const double preOut = _ReadWritePreDelay(in);

    LPTick(mInputLPState, preOut, bwCoef);
    double sig = mInputLPState;

    for (int i = 0; i < kInputAPs; i++)
      sig = AllpassTick(mInputAPBuf[i], mInputAPIdx[i], mInputAPLen[i], sig, inDiffCoefs[i]);

    // Read early-reflection short taps from the input AP chain output history. Reuse the
    // tank delay buffers as cheap per-sample short taps - we tap into mInputAPBuf[3] which
    // already holds the output of the fourth AP.
    double erL = 0.0;
    double erR = 0.0;
    if (erCount > 0)
    {
      auto& erBuf = mInputAPBuf[3];
      const size_t erBufSz = erBuf.size();
      for (int t = 0; t < erCount; t++)
      {
        const size_t tapFrames = ScaleDelay(mErTapMs[t], mSampleRate);
        const size_t safeFrames = std::min(tapFrames, erBufSz - 1);
        const size_t readIdx = (mInputAPIdx[3] + erBufSz - safeFrames) % erBufSz;
        const double tapVal = erBuf[readIdx] * mErTapGain[t];
        if (t % 2 == 0) erL += tapVal;
        else erR += tapVal;
      }
      // Normalise so adding ER doesn't blow the wet bus.
      const double norm = 1.0 / std::max(1, erCount / 2);
      erL *= norm * 0.5;
      erR *= norm * 0.5;
    }

    double tankInA = sig + mTank[1].lastOut * decayGain;
    double tankInB = sig + mTank[0].lastOut * decayGain;

    for (int h = 0; h < 2; h++)
    {
      double tankIn = (h == 0) ? tankInA : tankInB;

      double apOut = AllpassTick(mTank[h].apBuf, mTank[h].apIdx, tankApLen[h], tankIn, sm.decayApCoef);

      const double mod = std::sin(mPlateLfoPhase + h * kPI) * modDepth;
      double readPos = static_cast<double>(mTank[h].delIdx) - static_cast<double>(tankDelLen[h]) - mod;
      const size_t bufSz = mTank[h].delBuf.size();
      while (readPos < 0.0) readPos += bufSz;
      const size_t i0 = static_cast<size_t>(readPos) % bufSz;
      const size_t i1 = (i0 + 1) % bufSz;
      const double frac = readPos - std::floor(readPos);
      const double delOut = mTank[h].delBuf[i0] * (1.0 - frac) + mTank[h].delBuf[i1] * frac;

      mTank[h].delBuf[mTank[h].delIdx] = apOut;
      mTank[h].delIdx = (mTank[h].delIdx + 1) % bufSz;

      LPTick(mTank[h].lpState, delOut, dampCoef);
      mTank[h].lastOut = mTank[h].lpState;
    }

    const double outL = mTank[0].lastOut * 0.6 + mTank[1].lastOut * 0.4 + erL;
    const double outR = mTank[1].lastOut * 0.6 + mTank[0].lastOut * 0.4 + erR;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── TremVerb ──────────────────────────────────────────────────────

void Reverb::_ProcessTremVerb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  // Same Plate-derived pipeline but with shorter tanks (clamp internal effective decay) and
  // a photocell-shaped VCA on the wet bus only. Shimmer knob is repurposed as Trem Depth.
  const double inDiff1 = 0.75;
  const double inDiff2 = 0.625;
  const double inDiffCoefs[kInputAPs] = {inDiff1, inDiff1, inDiff2, inDiff2};

  // Short room/plate base: scale tank lengths down to a small-room range.
  const double lengthScale = 0.55;
  size_t tankDelLen[2];
  size_t tankApLen[2];
  for (int h = 0; h < 2; h++)
  {
    tankDelLen[h] = std::max<size_t>(1, static_cast<size_t>(mTank[h].delLen * lengthScale));
    tankDelLen[h] = std::min(tankDelLen[h], mTank[h].delBuf.size() - 4);
    tankApLen[h] = std::max<size_t>(1, static_cast<size_t>(mTank[h].apLen * lengthScale));
    tankApLen[h] = std::min(tankApLen[h], mTank[h].apBuf.size() - 16);
  }

  // Internal decay clamp at 2.5 s (anything beyond drowns the tremolo motion).
  const double effDecay = std::min(mDecay, 2.5);
  double decayGain = std::clamp(1.0 - (1.0 / (effDecay + 0.1)), 0.1, 0.93);

  const double cutoffHz = _ToneToCutoff(1800.0, 6000.0, 11000.0);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double dampCoef = dt / (rc + dt);

  const double bwCoef = 0.9995;
  const double modRate = 1.0 * 2.0 * kPI / mSampleRate;
  const double modDepth = 8.0 * mSampleRate / 29761.0;

  // Trem LFO: photocell-shaped (asymmetric duty). Implemented as a skewed sine.
  const double tremPhaseInc = mTremRateHz / mSampleRate;
  const double tremDepth = mShimmer; // Shimmer knob repurposed as Trem Depth (0..1).

  for (size_t s = 0; s < numFrames; s++)
  {
    mPlateLfoPhase += modRate;
    if (mPlateLfoPhase > 2.0 * kPI) mPlateLfoPhase -= 2.0 * kPI;

    mTremPhase += tremPhaseInc;
    if (mTremPhase >= 1.0) mTremPhase -= 1.0;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;
    const double preOut = _ReadWritePreDelay(in);

    LPTick(mInputLPState, preOut, bwCoef);
    double sig = mInputLPState;

    for (int i = 0; i < kInputAPs; i++)
      sig = AllpassTick(mInputAPBuf[i], mInputAPIdx[i], mInputAPLen[i], sig, inDiffCoefs[i]);

    double tankInA = sig + mTank[1].lastOut * decayGain;
    double tankInB = sig + mTank[0].lastOut * decayGain;

    for (int h = 0; h < 2; h++)
    {
      double tankIn = (h == 0) ? tankInA : tankInB;
      double apOut = AllpassTick(mTank[h].apBuf, mTank[h].apIdx, tankApLen[h], tankIn, 0.5);

      const double mod = std::sin(mPlateLfoPhase + h * kPI) * modDepth;
      double readPos = static_cast<double>(mTank[h].delIdx) - static_cast<double>(tankDelLen[h]) - mod;
      const size_t bufSz = mTank[h].delBuf.size();
      while (readPos < 0.0) readPos += bufSz;
      const size_t i0 = static_cast<size_t>(readPos) % bufSz;
      const size_t i1 = (i0 + 1) % bufSz;
      const double frac = readPos - std::floor(readPos);
      const double delOut = mTank[h].delBuf[i0] * (1.0 - frac) + mTank[h].delBuf[i1] * frac;

      mTank[h].delBuf[mTank[h].delIdx] = apOut;
      mTank[h].delIdx = (mTank[h].delIdx + 1) % bufSz;

      LPTick(mTank[h].lpState, delOut, dampCoef);
      mTank[h].lastOut = mTank[h].lpState;
    }

    const double outL = mTank[0].lastOut * 0.6 + mTank[1].lastOut * 0.4;
    const double outR = mTank[1].lastOut * 0.6 + mTank[0].lastOut * 0.4;

    // Photocell tremolo waveform: 0.5*(1 + sin(2*pi*phase)) shaped through tanh skew.
    const double rawSin = std::sin(2.0 * kPI * mTremPhase); // -1..1
    const double normSin = 0.5 * (rawSin + 1.0);             // 0..1
    // Skew toward asymmetric duty: tanh remap pulls the curve, with a small phase bias.
    const double skewed = 0.5 + 0.5 * std::tanh(2.5 * (normSin - 0.5));
    // Map to a VCA gain: 1 - depth + depth * skewed.
    const double tremGain = 1.0 - tremDepth + tremDepth * skewed;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wetRaw = (c == 0) ? outL : outR;
      const double wet = wetRaw * tremGain;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

} // namespace effect
} // namespace dsp
