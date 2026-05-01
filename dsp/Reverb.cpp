//
//  Reverb.cpp
//  VoLum - Hall (FDN), Plate (Dattorro), Oktaverb
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

// Scale a delay length from reference sample rate to actual
static size_t ScaleDelay(double ms, double sr) { return std::max<size_t>(1, static_cast<size_t>(ms * sr / 1000.0)); }
static size_t ScaleFromRef(size_t refLen, double refSR, double sr) { return std::max<size_t>(1, static_cast<size_t>(refLen * sr / refSR)); }
static double Hann(double phase) { return 0.5 - 0.5 * std::cos(kTwoPI * phase); }

// 1-pole lowpass tick: state = state*(1-c) + in*c; returns state
static double LPTick(double& state, double in, double coef) { state += coef * (in - state); return state; }

// Allpass tick (Moorer 2-multiply form): in-place buffer, returns output
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

Reverb::Reverb()
{
  mHallDelays.resize(kHallLines);
  mHallIndices.resize(kHallLines, 0);
  mHallLengths.resize(kHallLines, 0);
  mHallLPState.resize(kHallLines, 0.0);

  mInputAPBuf.resize(kInputAPs);
  mInputAPIdx.resize(kInputAPs, 0);
  mInputAPLen.resize(kInputAPs, 0);

  mOktPitchBufs.resize(kHallLines);
  mOktPitchWriteIdx.resize(kHallLines, 0);
  mOktPitchPhase.resize(kHallLines, 0.0);
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

void Reverb::SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode, double sampleRate)
{
  const bool srChanged = (mSampleRate != sampleRate);
  const double clampedPreDelayMs = std::clamp(preDelayMs, 0.0, 80.0);
  mSampleRate = sampleRate;
  mMix = std::clamp(mix, 0.0, 1.0);
  mDecay = std::clamp(decay, 0.1, 10.0);
  mTone = std::clamp(tone, 0.0, 10.0);
  _SetPreDelayLength(clampedPreDelayMs);
  mShimmer = std::clamp(shimmer, 0.0, 1.0);
  mMode = mode;

  if (srChanged) {
    mHallAllocated = false;
    mPlateAllocated = false;
    mOktaverbAllocated = false;
    _AllocatePreDelay();
  }
  else if (mPreDelayBuf.empty()) {
    _AllocatePreDelay();
  }

  if (mMode > 2) mMode = 0;
  if (mMode == 0 && !mHallAllocated) _AllocateHall();
  if (mMode == 1 && !mPlateAllocated) _AllocatePlate();
  if (mMode == 2) {
    if (!mHallAllocated) _AllocateHall();
    if (!mOktaverbAllocated) _AllocateOktaverb();
  }
}

void Reverb::_AllocatePreDelay()
{
  const size_t maxPreDelayLen = ScaleDelay(80.0, mSampleRate);
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
  const double baseLengthsMs[kHallLines] = { 31.0, 37.0, 43.0, 53.0, 61.0, 71.0, 79.0, 89.0 };
  for (int i = 0; i < kHallLines; i++) {
    mHallLengths[i] = ScaleDelay(baseLengthsMs[i], mSampleRate);
    size_t bufSz = mHallLengths[i] + 200;
    mHallDelays[i].assign(bufSz, 0.0);
    mHallIndices[i] = 0;
    mHallLPState[i] = 0.0;
  }
  mHallLfoPhase = 0.0;
  mHallAllocated = true;
}

void Reverb::_AllocateOktaverb()
{
  const size_t grainLen = std::max<size_t>(64, ScaleDelay(60.0, mSampleRate));
  for (int i = 0; i < kHallLines; i++) {
    mOktPitchBufs[i].assign(grainLen * 2 + 4, 0.0);
    mOktPitchWriteIdx[i] = 0;
    mOktPitchPhase[i] = 0.0;
  }
  mOktaverbAllocated = true;
}

void Reverb::_AllocatePlate()
{
  // Dattorro reference rate = 29761 Hz
  const double refSR = 29761.0;

  // Input diffusion allpass lengths (at reference rate)
  const size_t inputAPRef[kInputAPs] = { 142, 107, 379, 277 };
  for (int i = 0; i < kInputAPs; i++) {
    mInputAPLen[i] = ScaleFromRef(inputAPRef[i], refSR, mSampleRate);
    mInputAPBuf[i].assign(mInputAPLen[i] + 16, 0.0);
    mInputAPIdx[i] = 0;
  }

  // Tank half A: AP len=672, delay len=4453 (at ref rate)
  // Tank half B: AP len=908, delay len=4217
  const size_t tankAPRef[2] = { 672, 908 };
  const size_t tankDelRef[2] = { 4453, 4217 };

  for (int h = 0; h < 2; h++) {
    mTank[h].apLen = ScaleFromRef(tankAPRef[h], refSR, mSampleRate);
    mTank[h].apBuf.assign(mTank[h].apLen + 32, 0.0);
    mTank[h].apIdx = 0;

    mTank[h].delLen = ScaleFromRef(tankDelRef[h], refSR, mSampleRate);
    mTank[h].delBuf.assign(mTank[h].delLen + 32, 0.0);
    mTank[h].delIdx = 0;

    mTank[h].lpState = 0.0;
    mTank[h].lastOut = 0.0;
  }
  mInputLPState = 0.0;
  mPlateLfoPhase = 0.0;
  mPlateAllocated = true;
}

void Reverb::Reset()
{
  for (int i = 0; i < kHallLines; i++) {
    std::fill(mHallDelays[i].begin(), mHallDelays[i].end(), 0.0);
    mHallIndices[i] = 0;
    mHallLPState[i] = 0.0;
  }
  for (int i = 0; i < kInputAPs; i++) {
    std::fill(mInputAPBuf[i].begin(), mInputAPBuf[i].end(), 0.0);
    mInputAPIdx[i] = 0;
  }
  for (int h = 0; h < 2; h++) {
    std::fill(mTank[h].apBuf.begin(), mTank[h].apBuf.end(), 0.0);
    mTank[h].apIdx = 0;
    std::fill(mTank[h].delBuf.begin(), mTank[h].delBuf.end(), 0.0);
    mTank[h].delIdx = 0;
    mTank[h].lpState = 0.0;
    mTank[h].lastOut = 0.0;
  }
  std::fill(mPreDelayBuf.begin(), mPreDelayBuf.end(), 0.0);
  mPreDelayIdx = 0;
  for (int i = 0; i < kHallLines; i++) {
    std::fill(mOktPitchBufs[i].begin(), mOktPitchBufs[i].end(), 0.0);
    mOktPitchWriteIdx[i] = 0;
    mOktPitchPhase[i] = 0.0;
  }
  mInputLPState = 0.0;
  mHallLfoPhase = 0.0;
  mPlateLfoPhase = 0.0;
}

void Reverb::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  this->DSP::_PrepareBuffers(numChannels, numFrames);
}

DSP_SAMPLE** Reverb::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

  if (mMode == 2)
    _ProcessOktaverb(inputs, numChannels, numFrames);
  else if (mMode == 1)
    _ProcessPlate(inputs, numChannels, numFrames);
  else
    _ProcessHall(inputs, numChannels, numFrames);

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

double Reverb::_PitchDownOctaveTick(int line, double input)
{
  auto& buf = mOktPitchBufs[line];
  if (buf.empty())
    return input;

  size_t& writeIdx = mOktPitchWriteIdx[line];
  double& phase = mOktPitchPhase[line];
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
  phase += 0.5 / static_cast<double>(grainLen);
  if (phase >= 1.0)
    phase -= 1.0;

  return out;
}

// ─── Hall: improved 8-line FDN + Hadamard ─────────────────────────

void Reverb::_ProcessHall(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  // RT60 → loop gain
  const double avgDelay = 58.0 / 1000.0;
  double loopGain = std::pow(10.0, -3.0 * avgDelay / mDecay);
  loopGain = std::min(loopGain, 0.998);

  // Tone → LPF cutoff (mTone 0=dark, 10=bright)
  double cutoffHz = 800.0 + (mTone / 10.0) * 12000.0;
  double rc = 1.0 / (2.0 * kPI * cutoffHz);
  double dt = 1.0 / mSampleRate;
  double lpCoef = dt / (rc + dt);

  // Hadamard 8x8 (1/sqrt(8))
  const double H = 0.35355339;
  static const double mixMat[8][8] = {
    { H,  H,  H,  H,  H,  H,  H,  H},
    { H, -H,  H, -H,  H, -H,  H, -H},
    { H,  H, -H, -H,  H,  H, -H, -H},
    { H, -H, -H,  H,  H, -H, -H,  H},
    { H,  H,  H,  H, -H, -H, -H, -H},
    { H, -H,  H, -H, -H,  H, -H,  H},
    { H,  H, -H, -H, -H, -H,  H,  H},
    { H, -H, -H,  H, -H,  H,  H, -H}
  };

  // LFO rate: 0.6 Hz, depth: 10 samples
  const double lfoRate = 0.6 * 2.0 * kPI / mSampleRate;
  const double lfoDepth = 10.0;

  for (size_t s = 0; s < numFrames; s++)
  {
    mHallLfoPhase += lfoRate;
    if (mHallLfoPhase > 2.0 * kPI) mHallLfoPhase -= 2.0 * kPI;

    // Mono downmix
    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;
    in = _ReadWritePreDelay(in);

    // Read delay lines with modulation
    double readVals[8];
    for (int i = 0; i < kHallLines; i++)
    {
      double mod = std::sin(mHallLfoPhase + i * kPI * 0.25) * lfoDepth;
      double readPos = (double)mHallIndices[i] - (double)mHallLengths[i] - mod;
      size_t bufSz = mHallDelays[i].size();
      while (readPos < 0.0) readPos += bufSz;
      size_t i0 = (size_t)readPos % bufSz;
      size_t i1 = (i0 + 1) % bufSz;
      double frac = readPos - std::floor(readPos);
      double val = mHallDelays[i][i0] * (1.0 - frac) + mHallDelays[i][i1] * frac;

      LPTick(mHallLPState[i], val, lpCoef);
      readVals[i] = mHallLPState[i] * loopGain;
    }

    // Matrix feedback + write
    for (int i = 0; i < kHallLines; i++)
    {
      double fb = 0.0;
      for (int j = 0; j < kHallLines; j++) fb += mixMat[i][j] * readVals[j];
      mHallDelays[i][mHallIndices[i]] = in * 0.5 + fb;
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    // Stereo output: decorrelated taps
    double outL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5;
    double outR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5;

    for (size_t c = 0; c < numChannels; c++)
    {
      double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

void Reverb::_ProcessOktaverb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const double avgDelay = 58.0 / 1000.0;
  double loopGain = std::pow(10.0, -3.0 * avgDelay / mDecay);
  loopGain = std::min(loopGain, 0.998);

  const double cutoffHz = 800.0 + (mTone / 10.0) * 12000.0;
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double lpCoef = dt / (rc + dt);

  const double H = 0.35355339;
  static const double mixMat[8][8] = {
    { H,  H,  H,  H,  H,  H,  H,  H},
    { H, -H,  H, -H,  H, -H,  H, -H},
    { H,  H, -H, -H,  H,  H, -H, -H},
    { H, -H, -H,  H,  H, -H, -H,  H},
    { H,  H,  H,  H, -H, -H, -H, -H},
    { H, -H,  H, -H, -H,  H, -H,  H},
    { H,  H, -H, -H, -H, -H,  H,  H},
    { H, -H, -H,  H, -H,  H,  H, -H}
  };

  const double lfoRate = 0.6 * 2.0 * kPI / mSampleRate;
  const double lfoDepth = 10.0;
  const double shimmerOutput = mShimmer * 0.85;

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
      readVals[i] = mHallLPState[i] * loopGain;
      pitchedVals[i] = _PitchDownOctaveTick(i, readVals[i]);
    }

    for (int i = 0; i < kHallLines; i++)
    {
      double fb = 0.0;
      for (int j = 0; j < kHallLines; j++)
      {
        fb += mixMat[i][j] * readVals[j];
      }
      double write = in * 0.5 + fb;
      if (!std::isfinite(write)) { write = 0.0; Reset(); }
      mHallDelays[i][mHallIndices[i]] = std::clamp(write, -2.0, 2.0);
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    const double hallOutL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5;
    const double hallOutR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5;
    const double octaveOutL = (pitchedVals[0] + pitchedVals[2] - pitchedVals[4] + pitchedVals[6]) * 0.5;
    const double octaveOutR = (pitchedVals[1] - pitchedVals[3] + pitchedVals[5] + pitchedVals[7]) * 0.5;
    const double outL = hallOutL + octaveOutL * shimmerOutput;
    const double outR = hallOutR + octaveOutR * shimmerOutput;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── Plate: Dattorro allpass-loop reverb ──────────────────────────

void Reverb::_ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  // Input diffusion coefficients (from Dattorro paper)
  const double inDiff1 = 0.75;
  const double inDiff2 = 0.625;
  const double inDiffCoefs[kInputAPs] = { inDiff1, inDiff1, inDiff2, inDiff2 };

  // Decay → tank feedback gain
  double decayGain = std::clamp(1.0 - (1.0 / (mDecay + 0.1)), 0.1, 0.97);

  // Decay diffusion
  double decayDiff = 0.7;

  // Tone → damping LPF (same approach as hall)
  double cutoffHz = 1000.0 + (mTone / 10.0) * 14000.0;
  double rc = 1.0 / (2.0 * kPI * cutoffHz);
  double dt = 1.0 / mSampleRate;
  double dampCoef = dt / (rc + dt);

  // Input bandwidth LPF
  double bwCoef = 0.9995;

  // Tank modulation: ~1 Hz, excursion ~16 samples scaled to sample rate
  double modRate = 1.0 * 2.0 * kPI / mSampleRate;
  double modDepth = 16.0 * mSampleRate / 29761.0;

  for (size_t s = 0; s < numFrames; s++)
  {
    mPlateLfoPhase += modRate;
    if (mPlateLfoPhase > 2.0 * kPI) mPlateLfoPhase -= 2.0 * kPI;

    // Mono downmix
    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++) in += inputs[c][s];
    if (numChannels > 1) in *= 0.5;

    const double preOut = _ReadWritePreDelay(in);

    // Input bandwidth filter
    LPTick(mInputLPState, preOut, bwCoef);
    double sig = mInputLPState;

    // 4x input diffusion allpass
    for (int i = 0; i < kInputAPs; i++)
      sig = AllpassTick(mInputAPBuf[i], mInputAPIdx[i], mInputAPLen[i], sig, inDiffCoefs[i]);

    // Feed into tank (cross-coupled)
    double tankInA = sig + mTank[1].lastOut * decayGain;
    double tankInB = sig + mTank[0].lastOut * decayGain;

    // Process each tank half: AP → modulated delay → damping LP
    for (int h = 0; h < 2; h++)
    {
      double tankIn = (h == 0) ? tankInA : tankInB;

      // Decay diffusor allpass
      double apOut = AllpassTick(mTank[h].apBuf, mTank[h].apIdx, mTank[h].apLen, tankIn, decayDiff);

      // Modulated delay read
      double mod = std::sin(mPlateLfoPhase + h * kPI) * modDepth;
      double readPos = (double)mTank[h].delIdx - (double)mTank[h].delLen - mod;
      size_t bufSz = mTank[h].delBuf.size();
      while (readPos < 0.0) readPos += bufSz;
      size_t i0 = (size_t)readPos % bufSz;
      size_t i1 = (i0 + 1) % bufSz;
      double frac = readPos - std::floor(readPos);
      double delOut = mTank[h].delBuf[i0] * (1.0 - frac) + mTank[h].delBuf[i1] * frac;

      // Write AP output into delay
      mTank[h].delBuf[mTank[h].delIdx] = apOut;
      mTank[h].delIdx = (mTank[h].delIdx + 1) % bufSz;

      // Damping lowpass
      LPTick(mTank[h].lpState, delOut, dampCoef);
      mTank[h].lastOut = mTank[h].lpState;
    }

    // Output: taps from both tank halves for stereo decorrelation
    double outL = mTank[0].lastOut * 0.6 + mTank[1].lastOut * 0.4;
    double outR = mTank[1].lastOut * 0.6 + mTank[0].lastOut * 0.4;

    for (size_t c = 0; c < numChannels; c++)
    {
      double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] + wet * mMix;
      if (std::isnan(final_) || std::isinf(final_)) { final_ = 0.0; Reset(); }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

} // namespace effect
} // namespace dsp
