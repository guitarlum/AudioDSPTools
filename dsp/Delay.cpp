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

// Soft clip with mild asymmetry (used by the Analog feedback path).
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
  // Backward-compat 5-arg API: neutral defaults for the staging fields.
  SetParams(timeMs, feedback, mix, mode, sampleRate, /*tone*/ 0.5, /*age*/ 0.0,
            /*pingPong*/ false);
}

void Delay::SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate, double tone, double age,
                      bool pingPong)
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
      // alternate L<->R while decaying equally. The dry input is also cross-seeded so
      // a left-only impulse produces its first delayed repeat on the right.
      const double feedbackSrc = mPingPong ? readOther : readC;
      const double writeInput = (mPingPong && numChannels > 1 && c < 2) ? static_cast<double>(inputs[1 - c][s])
                                                                         : inputSample;
      mBuffer[c][mWriteIndex] = writeInput + feedbackSrc * mFeedback;
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
      const double readCBase = (c < 2) ? readBase[c] : ReadFractional(mBuffer[c], mWriteIndex, mCurrentDelayFrames, maxFrames);

      // Compander expansion on read (1:2 expand to recover dynamics).
      const double rectified = std::abs(readCBase);
      const double envCoeff = rectified > mCompandEnv ? compAttack : compRelease;
      mCompandEnv = envCoeff * mCompandEnv + (1.0 - envCoeff) * rectified;
      const double expandGain = 1.0 + mAge * std::clamp(mCompandEnv * 1.5, 0.0, 1.0);
      double readC = readCBase * expandGain;

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
      const double feedbackSrc = (mPingPong && numChannels > 1 && c < 2) ? readBase[1 - c] : readCBase;
      const double drive = 1.0 + std::max(0.0, mFeedback - 0.5) * 1.5;
      const double saturated = SoftClipAsym(feedbackSrc * mFeedback, drive);
      // Compander compression on write (gentle 2:1 around envelope).
      const double compressGain = 1.0 / (1.0 + mCompandEnv * 0.4);
      const double writeInput = (mPingPong && numChannels > 1 && c < 2) ? static_cast<double>(inputs[1 - c][s])
                                                                         : inputSample;
      mBuffer[c][mWriteIndex] = (writeInput + saturated) * compressGain;
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
      mReverseCaptureBuffer[c][mReverseIndex] = inputSample + reversedSample * mFeedback;
    }

    ++mReverseIndex;
  }

  return _GetPointers();
}

double Delay::_GetReverseFadeGain(size_t index, size_t segmentFrames) const
{
  const size_t fadeFrames = std::min<size_t>(64, segmentFrames / 2);
  if (fadeFrames == 0 || segmentFrames < 2)
    return 1.0;

  const size_t endDistance = segmentFrames - 1 - index;
  const size_t fadeDistance = std::min(index, endDistance);
  const double edgeFade = fadeDistance >= fadeFrames
                            ? 1.0
                            : static_cast<double>(fadeDistance) / static_cast<double>(fadeFrames);

  // Bloom keeps the old reverse-delay core at 0 and moves toward a softer whole-slice
  // sin^2 swell at 1, instead of replacing the old sound with a permanent triangle fade.
  const double t = static_cast<double>(index) / static_cast<double>(segmentFrames - 1);
  const double smoothBloom = std::sin(M_PI * t);
  const double sinSq = smoothBloom * smoothBloom;
  return edgeFade * (1.0 - mAge) + sinSq * mAge;
}

size_t Delay::_GetMaxFrames() const
{
  return std::max<size_t>(1, static_cast<size_t>(2.0 * mSampleRate));
}

} // namespace effect
} // namespace dsp
