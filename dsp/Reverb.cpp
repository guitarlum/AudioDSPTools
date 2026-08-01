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

static size_t ScaleDelay(double ms, double sr)
{
  return std::max<size_t>(1, static_cast<size_t>(ms * sr / 1000.0));
}
static size_t ScaleFromRef(size_t refLen, double refSR, double sr)
{
  return std::max<size_t>(1, static_cast<size_t>(refLen * sr / refSR));
}
static double Hann(double phase)
{
  return 0.5 - 0.5 * std::cos(kTwoPI * phase);
}

static double Hash01(unsigned int x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return static_cast<double>(x & 0x00ffffffU) / static_cast<double>(0x01000000U);
}

static double LPTick(double& state, double in, double coef)
{
  state += coef * (in - state);
  return state;
}

// Soft saturator used only by Oktaverb. Bounded in (-1, +1) for any finite input thanks
// to std::tanh; effectively transparent below ~0.3, gentle compression above, hard
// ceiling at unity. Catches DSP buildup (long decay, dual-pitch feedback, bloom envelope
// * wet gain) so the Oktaverb wet bus cannot rail. Hall and Plate intentionally do NOT
// run through this; the user explicitly liked the original additive Hall / Plate sound,
// and they never had the pitch-feedback runaway pathology that motivated the saturator.
static double SoftSaturate(double x)
{
  return std::tanh(x);
}

static double AllpassTick(std::vector<double>& buf, size_t& idx, size_t len, double input, double coef)
{
  if (len == 0 || buf.empty())
    return input;
  size_t bufSz = buf.size();
  size_t readIdx = (idx + bufSz - len) % bufSz;
  double delayed = buf[readIdx];
  double v = input + coef * delayed;
  buf[idx] = v;
  idx = (idx + 1) % bufSz;
  return delayed - coef * v;
}

// As above, but the read tap moves. Dattorro's plate modulates decay diffusion 1 by a
// few samples ("excursion") so the tank cannot settle into a fixed resonance.
static double ModulatedAllpassTick(std::vector<double>& buf, size_t& idx, size_t len, double input, double coef,
                                   double modSamples)
{
  if (len == 0 || buf.empty())
    return input;
  const size_t bufSz = buf.size();
  double readPos = static_cast<double>(idx) - static_cast<double>(len) - modSamples;
  while (readPos < 0.0)
    readPos += static_cast<double>(bufSz);
  const size_t i0 = static_cast<size_t>(readPos) % bufSz;
  const size_t i1 = (i0 + 1) % bufSz;
  const double frac = readPos - std::floor(readPos);
  const double delayed = buf[i0] * (1.0 - frac) + buf[i1] * frac;
  const double v = input + coef * delayed;
  buf[idx] = v;
  idx = (idx + 1) % bufSz;
  return delayed - coef * v;
}

// Read a circular buffer `offset` samples behind its write head.
static double ReadBufAt(const std::vector<double>& buf, size_t idx, size_t offset)
{
  if (buf.empty())
    return 0.0;
  const size_t sz = buf.size();
  return buf[(idx + sz - (offset % sz)) % sz];
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
  bool chorusedTail;
};

HallSubModeChar GetHallSubMode(int sub)
{
  switch (sub)
  {
    case 0: return {0.50, +1.5, 0.5, false}; // Studio
    case 2: return {1.70, -0.5, 1.5, true}; // Cathedral
    case 1:
    default: return {1.00, 0.0, 1.0, false}; // Concert
  }
}

constexpr double kOctaveUpRatio = 2.0;
constexpr double kOctaveDownRatio = 0.5;
constexpr double kSubFifthDownRatio = 0.6674199270850172; // -7 semitones

// Oktaverb sub-mode:
//   0 = Dark (-12 feedback + parallel sub-fifth)
//   1 = Shimmer (+12 feedback)
//   2 = Bloom (slow attack, no pitched feedback)
struct OktaverbSubModeChar
{
  // Primary feedback pitch voice (always single-voice in Shimmer; one half of the
  // dual pair in Halo). Bloom uses ratio 1.0 / gain 0.0 to disable.
  double feedbackPitchRatio;
  double feedbackPitchGain;
  // Optional secondary feedback pitch voice. Used by Halo to inject -12 alongside
  // the primary +12 so the tank is excited by both directions simultaneously.
  // gain == 0 disables the second voice (Shimmer / Bloom).
  double secondaryFeedbackPitchRatio;
  double secondaryFeedbackPitchGain;
  // Parallel pitch voice (added to wet output, never re-enters the tank). gain == 0
  // disables it. Currently unused by Halo and Shimmer; kept for future voicings.
  double parallelPitchRatio;
  double parallelPitchGain;
  double inputGain;
  double cutoffMinHz;
  double cutoffMidHz;
  double cutoffMaxHz;
  double modulationDepth;
  double wetWidth;
  double wetGain;
  bool bloom;
};

OktaverbSubModeChar GetOktaverbSubMode(int sub)
{
  switch (sub)
  {
    // Shimmer: bright FDN, +12 in feedback as the spine, no secondary voice, parallel +12
    // voice for body lift, wet boosted so Mix at ~30 percent reads over the dry amp signal.
    case 1:
      return {
        kOctaveUpRatio, 0.42, 1.0, 0.0, kOctaveUpRatio, 0.32, 0.75, 1200.0, 5000.0, 12000.0, 14.0, 1.10, 1.55, false};
    // Bloom: no pitch shifting, swell-driven envelope. It shares the same final output
    // safety as Halo/Shimmer so its default swell cannot clip when dry and wet align.
    case 2: return {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.50, 1200.0, 4700.0, 9500.0, 9.0, 1.25, 1.40, true};
    case 0:
    // Halo (Dual): both +12 and -12 in the feedback loop simultaneously (Valhalla Dual
    // / Meris Pitch Vector lineage). Bright FDN keeps the body audible; -12 brings the
    // doom-octave weight without burying the high-end. No parallel voice.
    default:
      return {
        kOctaveUpRatio, 0.30, kOctaveDownRatio, 0.30, 1.0, 0.0, 0.78, 1100.0, 4600.0, 11000.0, 11.0, 1.05, 1.55, false};
  }
}

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

  mFdnAPBuf.resize(kFdnDiffusers);
  mFdnAPIdx.resize(kFdnDiffusers, 0);
  mFdnAPLen.resize(kFdnDiffusers, 0);

  mFdnEarlyAPBuf.resize(kFdnEarlyAPs);
  mFdnEarlyAPIdx.resize(kFdnEarlyAPs, 0);
  mFdnEarlyAPLen.resize(kFdnEarlyAPs, 0);

  mPitchBufs.resize(kNumPitchVoices);
  mPitchWriteIdx.resize(kNumPitchVoices);
  mPitchPhase.resize(kNumPitchVoices);
  for (int v = 0; v < kNumPitchVoices; v++)
  {
    mPitchBufs[v].resize(kHallLines);
    mPitchWriteIdx[v].assign(kHallLines, 0);
    mPitchPhase[v].assign(kHallLines, 0.0);
  }

  mOktaverbLfoPhase.assign(kHallLines, 0.0);
  mOktaverbPitchLPState.assign(kHallLines, 0.0);
  mOktaverbFeedbackPitchLPState.assign(kHallLines, 0.0);
  mOktaverbFeedbackPitchHPState.assign(kHallLines, 0.0);
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

  if (!mHallAllocated)
    _AllocateHall();
  if (!mPlateAllocated)
    _AllocatePlate();
  if (!mOktaverbAllocated)
    _AllocateOktaverb();
}

void Reverb::SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode,
                       double sampleRate)
{
  SetParams(mix, decay, tone, preDelayMs, shimmer, mode, sampleRate, 0);
}

void Reverb::SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode,
                       double sampleRate, int subMode)
{
  const bool srChanged = (mSampleRate != sampleRate);
  const int prevMode = mMode;
  const int prevSubMode = mSubMode;
  // Keep the expanded pre-delay range for Oktaverb bloom.
  const double clampedPreDelayMs = std::clamp(preDelayMs, 0.0, 200.0);
  mSampleRate = sampleRate;
  mMix = std::clamp(mix, 0.0, 1.0);
  mDecay = std::clamp(decay, 0.1, 10.0);
  mTone = std::clamp(tone, 0.0, 10.0);
  _SetPreDelayLength(clampedPreDelayMs);
  mShimmer = std::clamp(shimmer, 0.0, 1.0);
  mMode = std::clamp(mode, 0, kNumModes - 1);
  mSubMode = std::clamp(subMode, 0, 2);

  // ~10 Hz one-pole towards the new Mix target. Inaudible delay (~16 ms time
  // constant) but kills the per-block stepping that DAW automation of the Mix
  // knob would otherwise produce on the equal-power dry/wet crossfade.
  if (mSampleRate > 0.0)
  {
    constexpr double kSmoothHz = 10.0;
    mMixSmoothCoef = 1.0 - std::exp(-2.0 * kPI * kSmoothHz / mSampleRate);
  }
  // On the very first SetParams from a fresh Reverb (or after an SR change that
  // already invalidated the algorithm allocations), snap the smoother to the new
  // target so the first block runs at the intended dry/wet mix instead of
  // ramping up from the constructor default.
  if (srChanged)
  {
    mMixSmoothed = mMix;
  }

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
    if (!mHallAllocated)
      _AllocateHall();
    if (!mOktaverbAllocated)
      _AllocateOktaverb();
  }

  // Hall (FDN), Plate (Dattorro), and Oktaverb share some allocations (kHallLines /
  // pitch lines) but each algorithm writes into its own indices. Switching between
  // modes without clearing state lets the previous algorithm's energy leak through
  // the new one for one decay tail length. Match Delay::SetParams which clears on
  // mode toggle, and also reset when the Oktaverb sub-mode (Halo / Shimmer / Bloom)
  // changes because each sub-mode owns distinct feedback paths.
  if (prevMode != mMode || (mMode == kModeOktaverb && prevSubMode != mSubMode))
    Reset();
}

void Reverb::_AllocatePreDelay()
{
  // Maximum pre-delay extended to 200 ms to support Oktaverb's pitched bloom.
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

  // Input diffuser. Mutually prime lengths so the nested allpasses do not reinforce
  // each other at a common period; the 0.75 / 0.625 coefficient pair is Dattorro's.
  const double fdnDiffuserMs[kFdnDiffusers] = {4.3, 6.7, 10.1, 14.9};
  for (int i = 0; i < kFdnDiffusers; i++)
  {
    mFdnAPLen[i] = ScaleDelay(fdnDiffuserMs[i], mSampleRate);
    mFdnAPBuf[i].assign(mFdnAPLen[i] + 16, 0.0);
    mFdnAPIdx[i] = 0;
  }

  // One more allpass per output channel, longer than any in the shared chain and
  // mutually prime with it, so left and right carry the same energy at different
  // instants. They inherit the chain's density; on their own a single long allpass
  // would just ring at multiples of its own delay.
  const double earlyAPMs[kFdnEarlyAPs] = {20.3, 27.7};
  for (int i = 0; i < kFdnEarlyAPs; i++)
  {
    mFdnEarlyAPLen[i] = ScaleDelay(earlyAPMs[i], mSampleRate);
    mFdnEarlyAPBuf[i].assign(mFdnEarlyAPLen[i] + 16, 0.0);
    mFdnEarlyAPIdx[i] = 0;
  }

  mHallAllocated = true;
}

void Reverb::_AllocateOktaverb()
{
  const size_t grainLen = std::max<size_t>(64, ScaleDelay(100.0, mSampleRate));
  for (int v = 0; v < kNumPitchVoices; v++)
  {
    for (int i = 0; i < kHallLines; i++)
    {
      mPitchBufs[v][i].assign(grainLen * 2 + 4, 0.0);
      mPitchWriteIdx[v][i] = 0;
      mPitchPhase[v][i] = 0.0;
    }
  }
  for (int i = 0; i < kHallLines; i++)
  {
    mOktaverbLfoPhase[i] = static_cast<double>(i) / kHallLines;
    mOktaverbPitchLPState[i] = 0.0;
    mOktaverbFeedbackPitchLPState[i] = 0.0;
    mOktaverbFeedbackPitchHPState[i] = 0.0;
  }
  mBloomEnv = 0.0;
  mBloomVCA = 0.0;
  mOktaverbAllocated = true;
}

void Reverb::_AllocatePlate()
{
  // Original Dattorro-style plate from dev. The iteration-2 Steel/Brass/Copper
  // experiment was intentionally removed from effect-staging.
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
  // Decay diffusion 2 and the second delay, absent through 1.2.0. They carry the
  // asymmetry between the halves: with them the loops are 10645 and 10944 samples,
  // without them both are 5125.
  const size_t tankAP2Ref[2] = {1800, 2656};
  const size_t tankDel2Ref[2] = {3720, 3163};

  for (int h = 0; h < 2; h++)
  {
    mTank[h].apLen = ScaleFromRef(tankAPRef[h], refSR, mSampleRate);
    mTank[h].apBuf.assign(mTank[h].apLen + 64, 0.0);
    mTank[h].apIdx = 0;

    mTank[h].delLen = ScaleFromRef(tankDelRef[h], refSR, mSampleRate);
    mTank[h].delBuf.assign(mTank[h].delLen + 32, 0.0);
    mTank[h].delIdx = 0;

    mTank[h].ap2Len = ScaleFromRef(tankAP2Ref[h], refSR, mSampleRate);
    mTank[h].ap2Buf.assign(mTank[h].ap2Len + 32, 0.0);
    mTank[h].ap2Idx = 0;

    mTank[h].del2Len = ScaleFromRef(tankDel2Ref[h], refSR, mSampleRate);
    mTank[h].del2Buf.assign(mTank[h].del2Len + 32, 0.0);
    mTank[h].del2Idx = 0;

    mTank[h].lpState = 0.0;
    mTank[h].lastOut = 0.0;
  }

  // Dattorro's output accumulators. Each channel reads four taps from the opposite
  // half and three from its own, which is where the stereo image comes from - not from
  // panning one half left and the other right, which is what 1.2.0 did.
  // Order per channel: delay1[a], delay1[b], ap2[c], delay2[d] of the far half, then
  // delay1[e], ap2[f], delay2[g] of the near half.
  const size_t tapLRef[kPlateOutTaps] = {266, 2974, 1913, 1996, 1990, 187, 1066};
  const size_t tapRRef[kPlateOutTaps] = {353, 3627, 1228, 2673, 2111, 335, 121};
  for (int t = 0; t < kPlateOutTaps; t++)
  {
    mPlateTapL[t] = ScaleFromRef(tapLRef[t], refSR, mSampleRate);
    mPlateTapR[t] = ScaleFromRef(tapRRef[t], refSR, mSampleRate);
  }

  mInputLPState = 0.0;
  mPlateLfoPhase = 0.0;
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
  for (int i = 0; i < kFdnDiffusers; i++)
  {
    std::fill(mFdnAPBuf[i].begin(), mFdnAPBuf[i].end(), 0.0);
    mFdnAPIdx[i] = 0;
  }
  for (int i = 0; i < kFdnEarlyAPs; i++)
  {
    std::fill(mFdnEarlyAPBuf[i].begin(), mFdnEarlyAPBuf[i].end(), 0.0);
    mFdnEarlyAPIdx[i] = 0;
  }
  for (int h = 0; h < 2; h++)
  {
    std::fill(mTank[h].apBuf.begin(), mTank[h].apBuf.end(), 0.0);
    mTank[h].apIdx = 0;
    std::fill(mTank[h].delBuf.begin(), mTank[h].delBuf.end(), 0.0);
    mTank[h].delIdx = 0;
    std::fill(mTank[h].ap2Buf.begin(), mTank[h].ap2Buf.end(), 0.0);
    mTank[h].ap2Idx = 0;
    std::fill(mTank[h].del2Buf.begin(), mTank[h].del2Buf.end(), 0.0);
    mTank[h].del2Idx = 0;
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
    mOktaverbLfoPhase[i] = static_cast<double>(i) / kHallLines;
    mOktaverbPitchLPState[i] = 0.0;
    mOktaverbFeedbackPitchLPState[i] = 0.0;
    mOktaverbFeedbackPitchHPState[i] = 0.0;
  }
  mBloomEnv = 0.0;
  mBloomVCA = 0.0;
  mInputLPState = 0.0;
  mHallLfoPhase = 0.0;
  mPlateLfoPhase = 0.0;
  mMixSmoothed = mMix;
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
    case kModeHall:
    default: _ProcessHall(inputs, numChannels, numFrames); break;
  }

  return _GetPointers();
}

// Nested allpass chain in front of the FDN. Each stage spreads an impulse into a
// decaying series of its own, so by the time the signal reaches the delay lines it is
// already a burst rather than a spike, and the eight line reads overlap into something
// continuous instead of arriving as eight separate events.
double Reverb::_DiffuseFdnInput(double input)
{
  static const double coefs[kFdnDiffusers] = {0.75, 0.75, 0.625, 0.625};
  double sig = input;
  for (int i = 0; i < kFdnDiffusers; i++)
    sig = AllpassTick(mFdnAPBuf[i], mFdnAPIdx[i], mFdnAPLen[i], sig, coefs[i]);
  return sig;
}

// The early field. Both channels are the diffuser output taken one allpass further,
// which spreads it over another few tens of milliseconds without touching its
// magnitude response, and differently for each channel.
void Reverb::_FdnEarlyField(double diffused, double& outL, double& outR)
{
  outL = AllpassTick(mFdnEarlyAPBuf[0], mFdnEarlyAPIdx[0], mFdnEarlyAPLen[0], diffused, 0.6);
  outR = AllpassTick(mFdnEarlyAPBuf[1], mFdnEarlyAPIdx[1], mFdnEarlyAPLen[1], diffused, 0.6);
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

  auto readFrac = [&](double grainPhase, double jitterFrames) {
    const double delay = 1.0 + grainPhase * static_cast<double>(grainLen) + jitterFrames;
    double readPos = static_cast<double>(writeIdx) - delay;
    while (readPos < 0.0)
      readPos += static_cast<double>(buf.size());
    const size_t i0 = static_cast<size_t>(readPos) % buf.size();
    const size_t i1 = (i0 + 1) % buf.size();
    const double frac = readPos - std::floor(readPos);
    return buf[i0] * (1.0 - frac) + buf[i1] * frac;
  };

  double sum = 0.0;
  double weight = 0.0;
  for (int grain = 0; grain < 3; ++grain)
  {
    double grainPhase = phase + static_cast<double>(grain) / 3.0;
    while (grainPhase >= 1.0)
      grainPhase -= 1.0;

    const double randA = Hash01(static_cast<unsigned int>((voice + 1) * 92821 + (line + 1) * 68917 + grain * 31337));
    const double randB = Hash01(static_cast<unsigned int>((voice + 5) * 19319 + (line + 3) * 49157 + grain * 27191));
    const double jitterFrames = randA * 0.008 * mSampleRate; // 0..8 ms splice offset
    const double detuneCents = (randB - 0.5) * 16.0; // +/-8 cents
    const double detuneRatio = std::pow(2.0, detuneCents / 1200.0);
    const double sample = readFrac(grainPhase, jitterFrames * detuneRatio);
    const double w = Hann(grainPhase);
    sum += sample * w;
    weight += w;
  }
  const double out = sum / std::max(0.000001, weight);

  writeIdx = (writeIdx + 1) % buf.size();
  // Moving the read delay shorter makes the read pointer travel faster (pitch up);
  // moving it longer makes the read pointer travel slower (pitch down).
  const double advance = std::abs(ratio - 1.0) / static_cast<double>(grainLen);
  if (ratio >= 1.0)
  {
    phase -= advance;
    while (phase < 0.0)
      phase += 1.0;
  }
  else
  {
    phase += advance;
    while (phase >= 1.0)
      phase -= 1.0;
  }

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
  // Same Cathedral-ish recipe as before - line lengths, loop gain, Hadamard matrix and
  // tone curve are untouched, so the tail is the one we had. What changed is how the
  // tank is excited and where the output is read from:
  //
  //   * the input runs through a nested allpass diffuser first, so each line receives a
  //     burst instead of a spike;
  //   * the lines are fed through a balanced sign pattern rather than all receiving the
  //     identical sample, so they stop reinforcing each other;
  //   * short taps read inside the lines, so wet energy exists before the shortest loop
  //     (52.7 ms) can produce any.
  //
  // Together those turn the first pass from eight audible clicks into a continuous
  // early field, and they take the hidden 52.7 ms floor out from under PRE-DLY.
  const HallSubModeChar sm = GetHallSubMode(2);

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

  const double cutoffBase = _ToneToCutoff(1500.0, 5000.0, 10000.0);
  const double cutoffHz = std::clamp(cutoffBase * std::pow(2.0, sm.cutoffOctaveOffset), 500.0, 18000.0);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double lpCoef = dt / (rc + dt);

  const double H = 0.35355339;
  static const double mixMat[8][8] = {{H, H, H, H, H, H, H, H},     {H, -H, H, -H, H, -H, H, -H},
                                      {H, H, -H, -H, H, H, -H, -H}, {H, -H, -H, H, H, -H, -H, H},
                                      {H, H, H, H, -H, -H, -H, -H}, {H, -H, H, -H, -H, H, -H, H},
                                      {H, H, -H, -H, -H, -H, H, H}, {H, -H, -H, H, -H, H, H, -H}};

  // Balanced +/- injection. Any zero-sum pattern works; what matters is that the lines
  // no longer see the same sample with the same sign.
  static const double injectSign[kHallLines] = {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0};

  const double lfoRate = 0.6 * 2.0 * kPI / mSampleRate;
  const double lfoDepth = 10.0 * sm.modDepthScale;

  const double cathRate = 0.2 / mSampleRate;
  const double cathDepth = sm.chorusedTail ? 4.0 : 0.0;

  // The early field is loud enough to close the gap without turning into a slapback of
  // its own; the tank still carries the body.
  const double earlyGain = 0.15;

  for (size_t s = 0; s < numFrames; s++)
  {
    mHallLfoPhase += lfoRate;
    if (mHallLfoPhase > 2.0 * kPI)
      mHallLfoPhase -= 2.0 * kPI;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++)
      in += inputs[c][s];
    if (numChannels > 1)
      in *= 0.5;
    in = _ReadWritePreDelay(in);
    const double diffused = _DiffuseFdnInput(in);

    double earlyL = 0.0;
    double earlyR = 0.0;
    _FdnEarlyField(diffused, earlyL, earlyR);

    double readVals[8];
    for (int i = 0; i < kHallLines; i++)
    {
      const double mod = std::sin(mHallLfoPhase + i * kPI * 0.25) * lfoDepth;
      double cathOffset = 0.0;
      if (sm.chorusedTail)
      {
        mHallChorusPhase[i] += cathRate;
        if (mHallChorusPhase[i] >= 1.0)
          mHallChorusPhase[i] -= 1.0;
        cathOffset = std::sin(2.0 * kPI * mHallChorusPhase[i]) * cathDepth;
      }

      double readPos = static_cast<double>(mHallIndices[i]) - static_cast<double>(effLengths[i]) - mod - cathOffset;
      const size_t bufSz = mHallDelays[i].size();
      while (readPos < 0.0)
        readPos += bufSz;
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
      for (int j = 0; j < kHallLines; j++)
        fb += mixMat[i][j] * readVals[j];
      double write = diffused * 0.5 * injectSign[i] + fb;
      if (!std::isfinite(write))
      {
        write = 0.0;
        Reset();
      }
      mHallDelays[i][mHallIndices[i]] = std::clamp(write, -3.0, 3.0);
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    const double outL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5 + earlyL * earlyGain;
    const double outR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5 + earlyR * earlyGain;

    // Equal-power crossfade. Pre-equal-power VoLum used `dry + wet*mMix` (additive),
    // which kept dry at unity at every Mix value and left reverb feeling "too quiet"
    // at musically normal settings. Equal-power matches Strymon BigSky / Eventide /
    // Neunaber convention: wet rises with sin(angle), dry falls with cos(angle), total
    // power preserved. kReverbWetTrim trims Hall/Plate up so they match Oktaverb
    // perceived loudness at Mix=0.5.
    mMixSmoothed += (mMix - mMixSmoothed) * mMixSmoothCoef;
    const double angle = mMixSmoothed * (kPI * 0.5);
    const double dryCoef = std::cos(angle);
    const double wetCoef = std::sin(angle) * kReverbWetTrim;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] * dryCoef + wet * wetCoef;
      if (std::isnan(final_) || std::isinf(final_))
      {
        final_ = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── Oktaverb ──────────────────────────────────────────────────────

void Reverb::_ProcessOktaverb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const OktaverbSubModeChar sm = GetOktaverbSubMode(mSubMode);

  // Oktaverb runs the same FDN as Hall at the unscaled length set, so it had the same
  // bare first pass - eight discrete taps between 31 and 89 ms - just less obvious,
  // because the taps sit closer together and the pitch grains smear everything after
  // the first generation. It takes Hall's input stage: diffuser, balanced injection and
  // early taps. Everything downstream of the tank write is unchanged.
  static const double injectSign[kHallLines] = {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0};
  const double earlyGain = 0.15;

  const double avgDelay = 58.0 / 1000.0;
  double loopGain = std::pow(10.0, -3.0 * avgDelay / mDecay);
  loopGain = std::min(loopGain, sm.bloom ? 0.996 : 0.985);

  const double cutoffHz = _ToneToCutoff(sm.cutoffMinHz, sm.cutoffMidHz, sm.cutoffMaxHz);
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double lpCoef = dt / (rc + dt);

  const double pitchCutoffHz = std::clamp(cutoffHz * (mSubMode == 0 ? 0.70 : 0.85), 120.0, 8500.0);
  const double pitchRc = 1.0 / (2.0 * kPI * pitchCutoffHz);
  const double pitchLpCoef = dt / (pitchRc + dt);
  // Halo (slot 0) carries the -12 voice and needs a real HP to prevent sub-bass build-up
  // in the FB loop; Shimmer (slot 1) is +12-only and wants a higher HP for clarity; Bloom
  // doesn't use the path so the value is moot.
  const double pitchHpCutoffHz = (mSubMode == 1) ? 180.0 : (mSubMode == 0) ? 120.0 : 45.0;
  const double pitchHpLpCoef = 1.0 - std::exp(-2.0 * kPI * pitchHpCutoffHz / mSampleRate);

  const double H = 0.35355339;
  static const double mixMat[8][8] = {{H, H, H, H, H, H, H, H},     {H, -H, H, -H, H, -H, H, -H},
                                      {H, H, -H, -H, H, H, -H, -H}, {H, -H, -H, H, H, -H, -H, H},
                                      {H, H, H, H, -H, -H, -H, -H}, {H, -H, H, -H, -H, H, -H, H},
                                      {H, H, -H, -H, -H, -H, H, H}, {H, -H, -H, H, -H, H, H, -H}};

  static const double lfoRatesHz[kHallLines] = {0.31, 0.43, 0.57, 0.71, 0.83, 0.97, 1.13, 1.29};
  const double intensity = std::clamp(mShimmer, 0.0, 1.0);
  // Sub-linear curve so the lower 50 percent of the knob already moves the pitch level
  // perceptibly; user feedback was that the knob "didn't seem to do too much" at the top.
  const double intensityCurved = std::pow(intensity, 0.7);
  const double feedbackPitchGain = sm.feedbackPitchGain * (0.10 + 0.90 * intensityCurved);
  const double secondaryFeedbackPitchGain = sm.secondaryFeedbackPitchGain * (0.10 + 0.90 * intensityCurved);
  // Parallel pitch always present at a small floor so the voicing colour reads even with
  // intensity at zero; Bloom / Halo keep parallel gain at zero via sm.parallelPitchGain.
  const double parallelPitchGain = sm.parallelPitchGain * (0.20 + 0.80 * intensityCurved);

  const double bloomAttackMs = 50.0 + 1950.0 * intensityCurved;
  const double bloomAttackCoef = 1.0 - std::exp(-1.0 / (std::max(1.0, bloomAttackMs) * 0.001 * mSampleRate));
  const double bloomReleaseCoef = 1.0 - std::exp(-1.0 / (1.8 * mSampleRate));
  const double bloomEnvAttackCoef = 1.0 - std::exp(-1.0 / (0.005 * mSampleRate));
  const double bloomEnvReleaseCoef = 1.0 - std::exp(-1.0 / (1.2 * mSampleRate));

  for (size_t s = 0; s < numFrames; s++)
  {
    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++)
      in += inputs[c][s];
    if (numChannels > 1)
      in *= 0.5;
    in = _ReadWritePreDelay(in);
    const double tankSource = _DiffuseFdnInput(in);

    double earlyL = 0.0;
    double earlyR = 0.0;
    _FdnEarlyField(tankSource, earlyL, earlyR);

    double readVals[8];
    double parallelPitchVals[8];
    for (int i = 0; i < kHallLines; i++)
    {
      const double rate = lfoRatesHz[i] * ((i & 1) ? 1.02 : 0.98);
      mOktaverbLfoPhase[i] += rate / mSampleRate;
      if (mOktaverbLfoPhase[i] >= 1.0)
        mOktaverbLfoPhase[i] -= 1.0;
      const double mod = std::sin(kTwoPI * mOktaverbLfoPhase[i]) * sm.modulationDepth;
      double readPos = static_cast<double>(mHallIndices[i]) - static_cast<double>(mHallLengths[i]) - mod;
      const size_t bufSz = mHallDelays[i].size();
      while (readPos < 0.0)
        readPos += bufSz;
      const size_t i0 = static_cast<size_t>(readPos) % bufSz;
      const size_t i1 = (i0 + 1) % bufSz;
      const double frac = readPos - std::floor(readPos);
      const double val = mHallDelays[i][i0] * (1.0 - frac) + mHallDelays[i][i1] * frac;

      LPTick(mHallLPState[i], val, lpCoef);
      const double feedback = mHallLPState[i] * loopGain;
      readVals[i] = feedback;

      parallelPitchVals[i] = 0.0;
      if (parallelPitchGain > 0.0)
      {
        const double pitched = _PitchShiftTick(kVoiceSubFifthParallel, i, feedback, sm.parallelPitchRatio);
        parallelPitchVals[i] = LPTick(mOktaverbPitchLPState[i], pitched, pitchLpCoef) * parallelPitchGain;
      }
    }

    const double absIn = std::abs(in);
    LPTick(mBloomEnv, absIn, absIn > mBloomEnv ? bloomEnvAttackCoef : bloomEnvReleaseCoef);
    const double bloomTarget = (mBloomEnv > 0.0001 || mBloomVCA > 0.0001) ? 1.0 : 0.0;
    LPTick(mBloomVCA, bloomTarget, bloomTarget > mBloomVCA ? bloomAttackCoef : bloomReleaseCoef);

    for (int i = 0; i < kHallLines; i++)
    {
      double fb = 0.0;
      for (int j = 0; j < kHallLines; j++)
        fb += mixMat[i][j] * readVals[j];
      if (feedbackPitchGain > 0.0 || secondaryFeedbackPitchGain > 0.0)
      {
        // Decorrelated source mixes per voice so Halo's two pitches don't share spectra.
        const double pitchSourcePrimary =
          0.25 * (readVals[i] + readVals[(i + 3) % kHallLines] - readVals[(i + 5) % kHallLines]);
        double pitched = 0.0;
        if (feedbackPitchGain > 0.0)
        {
          // Primary voice: +12 (Shimmer / Halo) or -12 (legacy Dark, no longer reachable).
          const double primary = _PitchShiftTick(kVoiceOctUpFeedback, i, pitchSourcePrimary, sm.feedbackPitchRatio);
          pitched += primary * feedbackPitchGain;
        }
        if (secondaryFeedbackPitchGain > 0.0)
        {
          // Secondary voice (Halo): -12 with its own decorrelated tap mix to keep the
          // dual character lush instead of phasey.
          const double pitchSourceSecondary =
            0.25 * (readVals[(i + 4) % kHallLines] - readVals[(i + 1) % kHallLines] + readVals[(i + 7) % kHallLines]);
          const double secondary =
            _PitchShiftTick(kVoiceOctDownFeedback, i, pitchSourceSecondary, sm.secondaryFeedbackPitchRatio);
          pitched += secondary * secondaryFeedbackPitchGain;
        }
        // LP and HP are linear so summing voices first then filtering matches per-voice
        // filtering; cheaper and shares state cleanly across modes.
        pitched = LPTick(mOktaverbFeedbackPitchLPState[i], pitched, pitchLpCoef);
        const double lowRumble = LPTick(mOktaverbFeedbackPitchHPState[i], pitched, pitchHpLpCoef);
        pitched -= lowRumble;
        fb += std::clamp(pitched, -0.65, 0.65);
      }
      double tankIn = sm.bloom ? tankSource : tankSource * sm.inputGain;
      tankIn *= injectSign[i];
      double write = tankIn + fb;
      if (!std::isfinite(write))
      {
        write = 0.0;
        Reset();
      }
      mHallDelays[i][mHallIndices[i]] = std::clamp(write, -2.0, 2.0);
      mHallIndices[i] = (mHallIndices[i] + 1) % mHallDelays[i].size();
    }

    const double center =
      (readVals[0] + readVals[1] + readVals[2] + readVals[3] + readVals[4] + readVals[5] + readVals[6] + readVals[7])
      * 0.125;
    const double hallTapL = (readVals[0] + readVals[2] - readVals[4] + readVals[6]) * 0.5;
    const double hallTapR = (readVals[1] - readVals[3] + readVals[5] + readVals[7]) * 0.5;
    const double hallOutL = center * (1.0 - sm.wetWidth) + hallTapL * sm.wetWidth;
    const double hallOutR = center * (1.0 - sm.wetWidth) + hallTapR * sm.wetWidth;
    const double pitchOutL =
      (parallelPitchVals[0] + parallelPitchVals[2] - parallelPitchVals[4] + parallelPitchVals[6]) * 0.5;
    const double pitchOutR =
      (parallelPitchVals[1] - parallelPitchVals[3] + parallelPitchVals[5] + parallelPitchVals[7]) * 0.5;
    const double bloomGain = sm.bloom ? mBloomVCA : 1.0;
    // The early field joins before the bloom VCA, so Bloom still swells from silence
    // rather than announcing itself with an early transient.
    const double outL = (hallOutL + pitchOutL + earlyL * earlyGain) * bloomGain * sm.wetGain;
    const double outR = (hallOutR + pitchOutR + earlyR * earlyGain) * bloomGain * sm.wetGain;

    // Safety stack scoped to Oktaverb:
    //   1. User's Mix knob internally scaled to 50 percent maximum, so the wet
    //      contribution can never sum to more than half-level on top of dry.
    //   2. tanh on the wet bus to keep wet-gain / FDN / pitch-feedback buildup
    //      bounded below +-1.0. Bloom skips this wet-bus tanh so the swell remains open.
    //   3. Final per-channel tanh on the dry+wet sum to prevent clipping. Bloom uses
    //      this same final shoulder when Mix is above zero; Mix=0 remains dry-pass.
    // Hall and Plate are untouched and use their original additive Mix in their
    // respective process functions.
    // Equal-power crossfade on the user-Mix angle. The 50% cap bounds the angle so user
    // mMix=1 maps to angle = pi/4 instead of bounding the wet coefficient. Oktaverb wet
    // bus already bakes sm.wetGain into the wet (1.40 / 1.55),
    // so no additional kReverbWetTrim is applied here. tanh saturators on wet (Halo /
    // Shimmer) and on the dry+wet sum are preserved.
    const double cap = 0.5;
    mMixSmoothed += (mMix - mMixSmoothed) * mMixSmoothCoef;
    const double angle = mMixSmoothed * cap * (kPI * 0.5);
    const double dryCoef = std::cos(angle);
    const double wetCoef = std::sin(angle);
    for (size_t c = 0; c < numChannels; c++)
    {
      const double rawWet = (c == 0) ? outL : outR;
      const double wet = sm.bloom ? rawWet : SoftSaturate(rawWet);
      double mixed = inputs[c][s] * dryCoef + wet * wetCoef;
      double final_ = (sm.bloom && mMix <= 0.0) ? mixed : SoftSaturate(mixed);
      if (std::isnan(final_) || std::isinf(final_))
      {
        final_ = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

// ─── Plate ─────────────────────────────────────────────────────────

void Reverb::_ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  // Dattorro's plate, now complete. Through 1.2.0 each tank half was one allpass and
  // one delay, and the output was a single tap off the end of each half. Two things
  // followed from that. Both halves came to exactly 5125 samples - 672 + 4453 and
  // 908 + 4217 - because the stages that carry the asymmetry were the missing ones, so
  // the halves rang in lockstep and produced one centred repeat every 172 ms. And with
  // no taps inside the lines there was nothing between those repeats, nor anything at
  // all before the first one.
  //
  // Restored here: decay diffusion 2 and the second delay per half, and the seven-tap
  // output accumulators. The tank is now 10645 and 10944 samples, and the taps read
  // from inside both halves, so the response is dense from the first millisecond.
  const double inDiff1 = 0.75;
  const double inDiff2 = 0.625;
  const double inDiffCoefs[kInputAPs] = {inDiff1, inDiff1, inDiff2, inDiff2};

  // Decay is now solved for the loop rather than approximated. The old
  // `1 - 1/(decay + 0.1)` was tuned against a tank half this size, so reusing it with
  // the full-length tank would have roughly doubled every decay time. Solving for RT60
  // keeps the knob honest in seconds - which is what its unit claims - and happens to
  // land within a few percent of the old curve through the middle of the range.
  const double tankSeconds = (mTank[0].apLen + mTank[0].delLen + mTank[0].ap2Len + mTank[0].del2Len + mTank[1].apLen
                              + mTank[1].delLen + mTank[1].ap2Len + mTank[1].del2Len)
                             / std::max(1.0, mSampleRate);
  // Per crossing, so one full trip through both halves attenuates by 60 dB after
  // mDecay seconds.
  double decayGain = std::pow(10.0, -3.0 * (tankSeconds * 0.5) / mDecay);
  decayGain = std::clamp(decayGain, 0.05, 0.98);

  const double decayDiff1 = 0.70;
  const double decayDiff2 = 0.50;

  double cutoffHz = 1000.0 + (mTone / 10.0) * 14000.0;
  const double rc = 1.0 / (2.0 * kPI * cutoffHz);
  const double dt = 1.0 / mSampleRate;
  const double dampCoef = dt / (rc + dt);

  const double bwCoef = 0.9995;

  // Excursion belongs on decay diffusion 1, per the reference; 1.2.0 modulated the
  // delay read instead.
  const double modRate = 1.0 * 2.0 * kPI / mSampleRate;
  const double modDepth = 16.0 * mSampleRate / 29761.0;

  // Dattorro's accumulator gain. The seven taps partly cancel, so this lands close to
  // the single-tap level the old path produced; kReverbWetTrim still applies on top.
  const double tapGain = 0.22;

  for (size_t s = 0; s < numFrames; s++)
  {
    mPlateLfoPhase += modRate;
    if (mPlateLfoPhase > 2.0 * kPI)
      mPlateLfoPhase -= 2.0 * kPI;

    double in = 0.0;
    for (size_t c = 0; c < numChannels; c++)
      in += inputs[c][s];
    if (numChannels > 1)
      in *= 0.5;
    const double preOut = _ReadWritePreDelay(in);

    LPTick(mInputLPState, preOut, bwCoef);
    double sig = mInputLPState;

    for (int i = 0; i < kInputAPs; i++)
      sig = AllpassTick(mInputAPBuf[i], mInputAPIdx[i], mInputAPLen[i], sig, inDiffCoefs[i]);

    const double tankIn[2] = {sig + mTank[1].lastOut * decayGain, sig + mTank[0].lastOut * decayGain};

    for (int h = 0; h < 2; h++)
    {
      TankHalf& t = mTank[h];
      const double mod = std::sin(mPlateLfoPhase + h * kPI) * modDepth;

      double x = ModulatedAllpassTick(t.apBuf, t.apIdx, t.apLen, tankIn[h], decayDiff1, mod);

      const size_t d1Sz = t.delBuf.size();
      const double d1 = t.delBuf[(t.delIdx + d1Sz - t.delLen) % d1Sz];
      t.delBuf[t.delIdx] = x;
      t.delIdx = (t.delIdx + 1) % d1Sz;

      LPTick(t.lpState, d1, dampCoef);

      x = AllpassTick(t.ap2Buf, t.ap2Idx, t.ap2Len, t.lpState, decayDiff2);

      const size_t d2Sz = t.del2Buf.size();
      const double d2 = t.del2Buf[(t.del2Idx + d2Sz - t.del2Len) % d2Sz];
      t.del2Buf[t.del2Idx] = x;
      t.del2Idx = (t.del2Idx + 1) % d2Sz;

      t.lastOut = d2;
    }

    // Four taps from the far half, three from the near one. This is where the stereo
    // image comes from; panning one half hard left and the other right, as 1.2.0 did,
    // cannot produce one when both halves are the same length.
    const TankHalf& a = mTank[0];
    const TankHalf& b = mTank[1];
    const double outL =
      tapGain
      * (ReadBufAt(b.delBuf, b.delIdx, mPlateTapL[0]) + ReadBufAt(b.delBuf, b.delIdx, mPlateTapL[1])
         - ReadBufAt(b.ap2Buf, b.ap2Idx, mPlateTapL[2]) + ReadBufAt(b.del2Buf, b.del2Idx, mPlateTapL[3])
         - ReadBufAt(a.delBuf, a.delIdx, mPlateTapL[4]) - ReadBufAt(a.ap2Buf, a.ap2Idx, mPlateTapL[5])
         - ReadBufAt(a.del2Buf, a.del2Idx, mPlateTapL[6]));
    const double outR =
      tapGain
      * (ReadBufAt(a.delBuf, a.delIdx, mPlateTapR[0]) + ReadBufAt(a.delBuf, a.delIdx, mPlateTapR[1])
         - ReadBufAt(a.ap2Buf, a.ap2Idx, mPlateTapR[2]) + ReadBufAt(a.del2Buf, a.del2Idx, mPlateTapR[3])
         - ReadBufAt(b.delBuf, b.delIdx, mPlateTapR[4]) - ReadBufAt(b.ap2Buf, b.ap2Idx, mPlateTapR[5])
         - ReadBufAt(b.del2Buf, b.del2Idx, mPlateTapR[6]));

    // Equal-power crossfade; see the Hall comment for the rationale.
    mMixSmoothed += (mMix - mMixSmoothed) * mMixSmoothCoef;
    const double angle = mMixSmoothed * (kPI * 0.5);
    const double dryCoef = std::cos(angle);
    const double wetCoef = std::sin(angle) * kReverbWetTrim;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double wet = (c == 0) ? outL : outR;
      double final_ = inputs[c][s] * dryCoef + wet * wetCoef;
      if (std::isnan(final_) || std::isinf(final_))
      {
        final_ = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(final_);
    }
  }
}

} // namespace effect
} // namespace dsp
