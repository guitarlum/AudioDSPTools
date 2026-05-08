//
//  Delay.cpp
//  VoLum
//

#include "Delay.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dsp
{
namespace effect
{

namespace
{

// Soft clip with mild asymmetry (used by Tape feedback path).
inline double SoftClipAsym(double x, double drive)
{
  const double y = std::tanh(drive * x + 0.05) - std::tanh(0.05);
  return y;
}

// Read a fractional sample from the ring buffer with linear interpolation.
inline double ReadFractional(const std::vector<double>& buf, size_t writeIndex, double delayFrames, size_t maxFrames)
{
  double readPos = static_cast<double>(writeIndex) - delayFrames;
  while (readPos < 0.0)
    readPos += maxFrames;
  while (readPos >= static_cast<double>(maxFrames))
    readPos -= maxFrames;

  size_t idx1 = static_cast<size_t>(readPos);
  size_t idx2 = (idx1 + 1) % maxFrames;
  double frac = readPos - static_cast<double>(idx1);
  return buf[idx1] * (1.0 - frac) + buf[idx2] * frac;
}

// Sub-mode tape character constants.
struct TapeChar
{
  double wowDepthCents;
  double wowRateHz;
  double flutterDepthCents;
  double flutterRateHz;
  double saturationDrive;
  double headBumpDb;
  double dropoutRateHz; // 0 == disabled
};

inline TapeChar GetTapeCharacter(int subMode, double age)
{
  // Sub-mode base values; Age scales most magnitudes.
  TapeChar tc{};
  switch (subMode)
  {
    case Delay::kTapeStudio:
      tc = {5.0, 0.4, 2.0, 6.0, 0.6, 1.5, 0.0};
      break;
    case Delay::kTapeBroken:
      tc = {15.0, 0.5, 4.0, 8.0, 1.6, 4.0, 0.7};
      break;
    case Delay::kTapeVintage:
    default:
      tc = {10.0, 0.6, 3.0, 7.0, 1.0, 3.0, 0.0};
      break;
  }
  // Age=0 halves modulation/saturation/dropouts; Age=1 amplifies.
  const double ageScale = 0.5 + age; // 0.5..1.5
  tc.wowDepthCents *= ageScale;
  tc.flutterDepthCents *= ageScale;
  tc.saturationDrive *= ageScale;
  tc.headBumpDb *= ageScale;
  if (tc.dropoutRateHz > 0.0)
    tc.dropoutRateHz *= ageScale;
  return tc;
}

} // namespace

Delay::Delay()
{
}

void Delay::Prepare(const size_t numChannels, const size_t numFrames, double sampleRate)
{
  mSampleRate = sampleRate;
  _PrepareBuffers(numChannels, numFrames);
}

void Delay::SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate)
{
  // Backward-compat 5-arg API: neutral defaults for the iteration-2 fields.
  SetParams(timeMs, feedback, mix, mode, sampleRate, /*tone*/ 0.5, /*age*/ 0.0,
            /*pingPong*/ false, /*tapeSubMode*/ kTapeVintage);
}

void Delay::SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate, double tone, double age,
                      bool pingPong, int tapeSubMode)
{
  if (mSampleRate != sampleRate)
  {
    mSampleRate = sampleRate;
    _PrepareDelayLines(mBuffer.size());
    _PrepareReverseBuffers(mReverseCaptureBuffer.size());
    mWriteIndex = 0;
    _ResetReverseState();
  }

  mTimeMs = std::clamp(timeMs, 10.0, 2000.0);
  mFeedback = std::clamp(feedback, 0.0, 0.99);
  mMix = std::clamp(mix, 0.0, 1.0);
  mMode = std::clamp(mode, 0, kNumModes - 1);
  mTone = std::clamp(tone, 0.0, 1.0);
  mAge = std::clamp(age, 0.0, 1.0);
  mPingPong = pingPong && mMode != kModeReverse;
  mTapeSubMode = std::clamp(tapeSubMode, 0, 2);

  mTargetDelayFrames = (mTimeMs / 1000.0) * mSampleRate;
  if (mCurrentDelayFrames == 0.0)
    mCurrentDelayFrames = mTargetDelayFrames;

  const size_t reverseSegmentFrames = std::clamp<size_t>(
    static_cast<size_t>(std::round(mTargetDelayFrames)), 1, _GetMaxFrames());
  if (reverseSegmentFrames != mReverseSegmentFrames)
  {
    mReverseSegmentFrames = reverseSegmentFrames;
    _ResetReverseState();
  }
}

void Delay::Reset()
{
  for (auto& buf : mBuffer)
    std::fill(buf.begin(), buf.end(), 0.0);
  mWriteIndex = 0;
  std::fill(mToneState.begin(), mToneState.end(), 0.0);
  std::fill(mFeedbackLpState.begin(), mFeedbackLpState.end(), 0.0);
  mWowPhase = 0.0;
  mFlutterPhase = 0.0;
  mChorusPhase = 0.0;
  mCompandEnv = 0.0;
  _ResetReverseState();
}

void Delay::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  this->DSP::_PrepareBuffers(numChannels, numFrames);
  _PrepareDelayLines(numChannels);
}

void Delay::_PrepareDelayLines(const size_t numChannels)
{
  const size_t maxFrames = _GetMaxFrames();
  if (mBuffer.size() != numChannels)
    mBuffer.resize(numChannels);

  for (auto& buf : mBuffer)
  {
    if (buf.size() != maxFrames)
      buf.assign(maxFrames, 0.0);
  }

  if (mToneState.size() != numChannels)
    mToneState.assign(numChannels, 0.0);
  if (mFeedbackLpState.size() != numChannels)
    mFeedbackLpState.assign(numChannels, 0.0);
}

void Delay::_PrepareReverseBuffers(const size_t numChannels)
{
  const size_t maxFrames = _GetMaxFrames();
  bool resized = false;

  if (mReverseCaptureBuffer.size() != numChannels)
  {
    mReverseCaptureBuffer.resize(numChannels);
    mReversePlaybackBuffer.resize(numChannels);
    resized = true;
  }

  for (size_t c = 0; c < numChannels; ++c)
  {
    if (mReverseCaptureBuffer[c].size() != maxFrames)
    {
      mReverseCaptureBuffer[c].assign(maxFrames, 0.0);
      resized = true;
    }
    if (mReversePlaybackBuffer[c].size() != maxFrames)
    {
      mReversePlaybackBuffer[c].assign(maxFrames, 0.0);
      resized = true;
    }
  }

  if (resized)
    _ResetReverseState();
}

void Delay::_ResetReverseState()
{
  for (auto& buf : mReverseCaptureBuffer)
    std::fill(buf.begin(), buf.end(), 0.0);
  for (auto& buf : mReversePlaybackBuffer)
    std::fill(buf.begin(), buf.end(), 0.0);
  mReverseIndex = 0;
  mReversePlaybackReady = false;
}

double Delay::_ApplyToneTilt(size_t channel, double sample, double tone, double cutoffHz)
{
  if (channel >= mToneState.size())
    return sample;
  if (mSampleRate <= 0.0)
    return sample;

  // One-pole tilt: y = mix(highpassed, lowpassed, tone)
  const double rc = 1.0 / (2.0 * M_PI * std::clamp(cutoffHz, 50.0, 16000.0));
  const double dt = 1.0 / mSampleRate;
  const double alpha = dt / (rc + dt);
  mToneState[channel] += alpha * (sample - mToneState[channel]);
  const double low = mToneState[channel];
  const double high = sample - low;
  // tone == 0.5: pass-through; <0.5 darker (more low); >0.5 brighter (more high)
  const double mix = (tone - 0.5) * 2.0; // -1..1
  if (mix >= 0.0)
    return low + high * (1.0 + mix * 0.5); // boost high up to +50%
  return low * (1.0 - mix * 0.5) + high * (1.0 + mix); // shelf low up
}

DSP_SAMPLE** Delay::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

  switch (mMode)
  {
    case kModeReverse: return _ProcessReverse(inputs, numChannels, numFrames);
    case kModeAnalog: return _ProcessAnalog(inputs, numChannels, numFrames);
    case kModeTape: return _ProcessTape(inputs, numChannels, numFrames);
    case kModeDigital:
    default: return _ProcessDigital(inputs, numChannels, numFrames);
  }
}

DSP_SAMPLE** Delay::_ProcessDigital(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const size_t maxFrames = _GetMaxFrames();
  // Tone cutoff anchored around 4 kHz; tone moves perceived brightness via the helper.
  const double toneCutoff = 4000.0;

  // Age (Digital) = bit-crush + low-level noise floor. age=0 -> bit-perfect.
  const int crushBits = mAge > 0.0 ? std::max(8, 16 - static_cast<int>(std::round(mAge * 6.0))) : 24;
  const double crushStep = std::pow(2.0, -static_cast<double>(crushBits - 1));
  const double noiseAmp = mAge > 0.0 ? mAge * 1.0e-3 : 0.0;
  unsigned int noiseSeed = 12345;
  auto rnd = [&]() {
    noiseSeed = noiseSeed * 1103515245u + 12345u;
    return (static_cast<double>((noiseSeed >> 8) & 0xFFFF) / 65535.0 - 0.5) * 2.0;
  };

  for (size_t s = 0; s < numFrames; s++)
  {
    mCurrentDelayFrames += 0.001 * (mTargetDelayFrames - mCurrentDelayFrames);

    // Read both channels first so ping-pong cross-feed is symmetric.
    double readSample[2] = {0.0, 0.0};
    for (size_t c = 0; c < numChannels && c < 2; c++)
      readSample[c] = ReadFractional(mBuffer[c], mWriteIndex, mCurrentDelayFrames, maxFrames);

    for (size_t c = 0; c < numChannels; c++)
    {
      const double inputSample = inputs[c][s];
      const double readC = (c < 2) ? readSample[c] : 0.0;
      const double readOther = (numChannels > 1 && c < 2) ? readSample[1 - c] : readC;

      // Wet path: tone tilt on the read tap.
      double wet = _ApplyToneTilt(c, readC, mTone, toneCutoff);

      // Age: bit-crush + noise on the wet line.
      if (mAge > 0.0)
      {
        wet = std::round(wet / crushStep) * crushStep;
        wet += noiseAmp * rnd();
      }

      double finalSample = inputSample + wet * mMix;
      if (!std::isfinite(finalSample))
      {
        finalSample = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);

      // Ping-pong: feed the OPPOSITE channel's read tap into our write tap so repeats
      // alternate L<->R while decaying equally.
      const double feedbackSrc = mPingPong ? readOther : readC;
      mBuffer[c][mWriteIndex] = inputSample + feedbackSrc * mFeedback;
    }

    mWriteIndex = (mWriteIndex + 1) % maxFrames;
  }

  return _GetPointers();
}

DSP_SAMPLE** Delay::_ProcessAnalog(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const size_t maxFrames = _GetMaxFrames();

  // BBD-style: per-repeat HF rolloff anchored to delay time.
  // Cutoff drops with delay time and slightly more with Age.
  const double timeFactor = std::clamp(mTimeMs / 600.0, 0.1, 1.5);
  const double bbdCutoff = std::clamp(7000.0 / (1.0 + timeFactor * (1.0 + mAge)), 1500.0, 7000.0);

  // Optical chorus on the wet bus.
  const double chorusRateHz = 0.4 + (1.0 - mAge) * 0.4; // slower at high Age
  const double chorusDepthMs = 3.0 + mAge * 7.0;
  const double chorusDepthFrames = (chorusDepthMs / 1000.0) * mSampleRate;
  const double chorusCenterFrames = (4.0 / 1000.0) * mSampleRate;

  // Compander parameters (envelope follower).
  const double compAttack = std::exp(-1.0 / (0.005 * mSampleRate));
  const double compRelease = std::exp(-1.0 / (0.150 * mSampleRate));

  for (size_t s = 0; s < numFrames; s++)
  {
    mCurrentDelayFrames += 0.001 * (mTargetDelayFrames - mCurrentDelayFrames);

    // LFO tick.
    mChorusPhase += chorusRateHz / mSampleRate;
    if (mChorusPhase >= 1.0)
      mChorusPhase -= 1.0;
    const double chorusOffsetFrames = chorusCenterFrames + chorusDepthFrames * 0.5 * std::sin(2.0 * M_PI * mChorusPhase);

    double readBase[2] = {0.0, 0.0};
    for (size_t c = 0; c < numChannels && c < 2; c++)
      readBase[c] = ReadFractional(mBuffer[c], mWriteIndex, mCurrentDelayFrames, maxFrames);

    for (size_t c = 0; c < numChannels; c++)
    {
      const double inputSample = inputs[c][s];

      // Compander expansion on read (1:2 expand to recover dynamics).
      const double rectified = std::abs(readBase[c]);
      const double envCoeff = rectified > mCompandEnv ? compAttack : compRelease;
      mCompandEnv = envCoeff * mCompandEnv + (1.0 - envCoeff) * rectified;
      const double expandGain = 1.0 + mAge * std::clamp(mCompandEnv * 1.5, 0.0, 1.0);
      double readC = readBase[c] * expandGain;

      // Per-repeat BBD-style LP in the feedback path (one-pole).
      const double rc = 1.0 / (2.0 * M_PI * bbdCutoff);
      const double dt = 1.0 / mSampleRate;
      const double alpha = dt / (rc + dt);
      mFeedbackLpState[c] += alpha * (readC - mFeedbackLpState[c]);
      const double bbdRead = mFeedbackLpState[c];

      // Optical chorus on the wet line: a second tap with LFO-modulated offset.
      const double chorusRead = ReadFractional(mBuffer[c], mWriteIndex, mCurrentDelayFrames + chorusOffsetFrames, maxFrames);
      double wet = bbdRead * 0.6 + chorusRead * 0.4;

      // Tone tilt on wet bus.
      wet = _ApplyToneTilt(c, wet, mTone, 3500.0);

      double finalSample = inputSample + wet * mMix;
      if (!std::isfinite(finalSample))
      {
        finalSample = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);

      // Soft saturation in feedback path; intensity rises with feedback.
      const double feedbackSrc = mPingPong && numChannels > 1 ? readBase[1 - c] : readBase[c];
      const double drive = 1.0 + std::max(0.0, mFeedback - 0.5) * 1.5;
      const double saturated = SoftClipAsym(feedbackSrc * mFeedback, drive);
      // Compander compression on write (gentle 2:1 around envelope).
      const double compressGain = 1.0 / (1.0 + mCompandEnv * 0.4);
      mBuffer[c][mWriteIndex] = (inputSample + saturated) * compressGain;
    }

    mWriteIndex = (mWriteIndex + 1) % maxFrames;
  }

  return _GetPointers();
}

DSP_SAMPLE** Delay::_ProcessTape(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  const size_t maxFrames = _GetMaxFrames();
  const TapeChar tc = GetTapeCharacter(mTapeSubMode, mAge);

  // Per-repeat HF rolloff (tape head wear / azimuth approximation).
  const double tapeCutoff = std::clamp(7000.0 - mAge * 3500.0, 1500.0, 7000.0);

  // Dropout LFO (only when Broken sub-mode and age > 0.3).
  const double dropoutEnabled = (tc.dropoutRateHz > 0.0 && mAge > 0.3) ? 1.0 : 0.0;
  static double sDropoutPhase = 0.0;

  // Frames-of-cents conversion: 1 cent of pitch = 1/1200 octave; for a small detune we
  // approximate via a slow read-tap modulation around mCurrentDelayFrames.
  // Convert cents to fraction of a frame at the local sample rate. Scale empirically to
  // match perceived wow depth at typical delay times.
  const double centsToFracPerSec = 0.001; // tunable

  for (size_t s = 0; s < numFrames; s++)
  {
    mCurrentDelayFrames += 0.001 * (mTargetDelayFrames - mCurrentDelayFrames);

    // Wow + flutter LFOs.
    mWowPhase += tc.wowRateHz / mSampleRate;
    if (mWowPhase >= 1.0)
      mWowPhase -= 1.0;
    mFlutterPhase += tc.flutterRateHz / mSampleRate;
    if (mFlutterPhase >= 1.0)
      mFlutterPhase -= 1.0;

    sDropoutPhase += tc.dropoutRateHz / mSampleRate;
    if (sDropoutPhase >= 1.0)
      sDropoutPhase -= 1.0;

    const double wowSamples = (tc.wowDepthCents * centsToFracPerSec) * mSampleRate
                              * std::sin(2.0 * M_PI * mWowPhase);
    const double flutterSamples = (tc.flutterDepthCents * centsToFracPerSec) * mSampleRate
                                  * std::sin(2.0 * M_PI * mFlutterPhase);

    // Dropout envelope: occasional notch when sub-mode = Broken at high age.
    const double dropoutRaw = std::sin(2.0 * M_PI * sDropoutPhase);
    const double dropoutGain = 1.0 - dropoutEnabled * std::max(0.0, dropoutRaw - 0.85) * 4.0;

    for (size_t c = 0; c < numChannels; c++)
    {
      const double inputSample = inputs[c][s];

      // Per-channel quarter-cycle decorrelation.
      const double channelOffset = (numChannels > 1 && c == 1) ? 0.25 : 0.0;
      const double wowSamplesC = (tc.wowDepthCents * centsToFracPerSec) * mSampleRate
                                 * std::sin(2.0 * M_PI * (mWowPhase + channelOffset));
      const double flutterSamplesC = (tc.flutterDepthCents * centsToFracPerSec) * mSampleRate
                                     * std::sin(2.0 * M_PI * (mFlutterPhase + channelOffset));
      (void) wowSamples;
      (void) flutterSamples;

      const double readDelay = mCurrentDelayFrames + wowSamplesC + flutterSamplesC;
      double readC = ReadFractional(mBuffer[c], mWriteIndex, readDelay, maxFrames);

      // Per-repeat HF rolloff.
      const double rc = 1.0 / (2.0 * M_PI * tapeCutoff);
      const double dt = 1.0 / mSampleRate;
      const double alpha = dt / (rc + dt);
      mFeedbackLpState[c] += alpha * (readC - mFeedbackLpState[c]);
      double wet = mFeedbackLpState[c];

      // Apply dropout to wet only.
      wet *= dropoutGain;

      // Tone tilt.
      wet = _ApplyToneTilt(c, wet, mTone, 3500.0);

      double finalSample = inputSample + wet * mMix;
      if (!std::isfinite(finalSample))
      {
        finalSample = 0.0;
        Reset();
      }
      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);

      // Saturate the feedback signal; ping-pong cross-feed if enabled.
      const double feedbackSrc = mPingPong && numChannels > 1
                                   ? ReadFractional(mBuffer[1 - c], mWriteIndex, readDelay, maxFrames)
                                   : readC;
      const double driven = SoftClipAsym(feedbackSrc * mFeedback, 1.0 + tc.saturationDrive);
      // Head bump: a tiny boost around 100 Hz approximated as a second-order resonance is
      // overkill; use a small fixed mid-low shelf on the feedback bus.
      const double bumpGain = std::pow(10.0, tc.headBumpDb * 0.05);
      mBuffer[c][mWriteIndex] = inputSample * bumpGain + driven;
    }

    mWriteIndex = (mWriteIndex + 1) % maxFrames;
  }

  return _GetPointers();
}

DSP_SAMPLE** Delay::_ProcessReverse(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareReverseBuffers(numChannels);

  const size_t segmentFrames = std::clamp(mReverseSegmentFrames, static_cast<size_t>(1), _GetMaxFrames());

  for (size_t s = 0; s < numFrames; s++)
  {
    if (mReverseIndex >= segmentFrames)
    {
      std::swap(mReverseCaptureBuffer, mReversePlaybackBuffer);
      for (auto& buf : mReverseCaptureBuffer)
        std::fill_n(buf.begin(), segmentFrames, 0.0);
      mReverseIndex = 0;
      mReversePlaybackReady = true;
    }

    const size_t playbackIndex = segmentFrames - 1 - mReverseIndex;
    const double fadeGain = _GetReverseFadeGain(mReverseIndex, segmentFrames);

    for (size_t c = 0; c < numChannels; c++)
    {
      const double inputSample = inputs[c][s];
      double reversedSample = mReversePlaybackReady ? mReversePlaybackBuffer[c][playbackIndex] * fadeGain : 0.0;
      // Tone tilt on wet only.
      reversedSample = _ApplyToneTilt(c, reversedSample, mTone, 3500.0);

      double finalSample = inputSample * (1.0 - mMix) + reversedSample * mMix;

      if (!std::isfinite(finalSample))
      {
        finalSample = 0.0;
        Reset();
      }

      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);
      // Internal feedback clamp: reverse self-feedback above ~0.85 reads as chaos.
      const double effFeedback = std::min(mFeedback, 0.85);
      mReverseCaptureBuffer[c][mReverseIndex] = inputSample + reversedSample * effFeedback;
    }

    ++mReverseIndex;
  }

  return _GetPointers();
}

double Delay::_GetReverseFadeGain(size_t index, size_t segmentFrames) const
{
  if (segmentFrames < 2)
    return 0.0;
  // Age picks the fade-shape softness:
  //   0.0 -> sharp triangle, 0.5 -> half-sine, 1.0 -> sin^2 (smooth swell).
  const double t = static_cast<double>(index) / static_cast<double>(segmentFrames - 1); // 0..1

  // Triangle: linear ramp up then down.
  const double tri = (t < 0.5) ? (t * 2.0) : ((1.0 - t) * 2.0);
  // Half-sine: sin(pi*t).
  const double halfSin = std::sin(M_PI * t);
  // sin^2: smoother bloom.
  const double sinSq = halfSin * halfSin;

  // Blend between triangle, half-sine, sin^2 according to age (piecewise linear).
  if (mAge < 0.5)
  {
    const double w = mAge * 2.0; // 0..1
    return tri * (1.0 - w) + halfSin * w;
  }
  const double w = (mAge - 0.5) * 2.0; // 0..1
  return halfSin * (1.0 - w) + sinSq * w;
}

size_t Delay::_GetMaxFrames() const
{
  return std::max<size_t>(1, static_cast<size_t>(2.0 * mSampleRate));
}

} // namespace effect
} // namespace dsp
