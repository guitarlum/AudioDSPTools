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
  const int prevMode = mMode;
  const bool prevPingPong = mPingPong;

  if (mSampleRate != sampleRate)
  {
    mSampleRate = sampleRate;
    _PrepareDelayLines(mBuffer.size());
    _PrepareReverseBuffers(mReverseRing.size());
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

  // Overlap-add reverse: in-flight voices keep their original length when this
  // changes, so updating segment frames here is glitch-free (no Reset needed).
  // Cap to half the ring so two voices can coexist without wrap-around aliasing
  // into the in-flight voice's snapshot. Must be set BEFORE Reset() below so the
  // post-reset launch countdown reflects the new slice length.
  mReverseSegmentFrames = std::clamp<size_t>(
    static_cast<size_t>(std::round(mTargetDelayFrames)), 2,
    std::max<size_t>(2, _GetMaxFrames() / 2));

  if (prevMode != mMode || prevPingPong != mPingPong)
    Reset();
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
  const size_t ringSize = _GetMaxFrames();
  bool resized = false;

  if (mReverseRing.size() != numChannels)
  {
    mReverseRing.resize(numChannels);
    resized = true;
  }
  for (size_t c = 0; c < numChannels; ++c)
  {
    if (mReverseRing[c].size() != ringSize)
    {
      mReverseRing[c].assign(ringSize, 0.0);
      resized = true;
    }
  }
  mReverseRingSize = ringSize;

  if (resized)
    _ResetReverseState();
}

void Delay::_ResetReverseState()
{
  for (auto& buf : mReverseRing)
    std::fill(buf.begin(), buf.end(), 0.0);
  mReverseWritePos = 0;
  // First voice can launch only after one full slice has been captured; otherwise
  // it would read silence (or stale ring contents) and produce a "tick" at start.
  mReverseFramesUntilLaunch = std::max<size_t>(1, mReverseSegmentFrames);
  for (auto& v : mReverseVoices)
    v = ReverseVoice{};
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

      // Ping-pong: opposite-channel read into feedback; seed right delay line only so L=R mono
      // still produces R-first alternating repeats (cross-feed fills left on later taps).
      double feedbackSrc;
      double writeInput;
      if (mPingPong && numChannels > 1 && c < 2)
      {
        feedbackSrc = readOther;
        writeInput = (c == 1) ? static_cast<double>(inputs[0][s]) : 0.0;
      }
      else
      {
        feedbackSrc = readC;
        writeInput = inputSample;
      }
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
      double feedbackSrcBase;
      double writeInput;
      if (mPingPong && numChannels > 1 && c < 2)
      {
        feedbackSrcBase = readBase[1 - c];
        writeInput = (c == 1) ? static_cast<double>(inputs[0][s]) : 0.0;
      }
      else
      {
        feedbackSrcBase = readCBase;
        writeInput = inputSample;
      }
      const double drive = 1.0 + std::max(0.0, mFeedback - 0.5) * 1.5;
      const double saturated = SoftClipAsym(feedbackSrcBase * mFeedback, drive);
      // Compander compression on write (gentle 2:1 around envelope).
      const double compressGain = 1.0 / (1.0 + mCompandEnv * 0.4);
      mBuffer[c][mWriteIndex] = (writeInput + saturated) * compressGain;
    }

    mWriteIndex = (mWriteIndex + 1) % maxFrames;
  }

  return _GetPointers();
}

DSP_SAMPLE** Delay::_ProcessReverse(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareReverseBuffers(numChannels);

  const size_t ringSize = mReverseRingSize;
  if (ringSize == 0 || numChannels == 0)
    return _GetPointers();

  const size_t segmentFrames = std::clamp<size_t>(mReverseSegmentFrames, 2, ringSize / 2);

  for (size_t s = 0; s < numFrames; s++)
  {
    // 1) Launch a new voice when the stagger countdown expires. Two voices alternate;
    //    a fresh launch picks the inactive slot (or steals the older one). The
    //    countdown uses the *current* segmentFrames, so time-knob changes fade in
    //    cleanly on the next launch without disturbing in-flight voices.
    if (mReverseFramesUntilLaunch == 0)
    {
      int slot = -1;
      for (int v = 0; v < 2; ++v)
      {
        if (!mReverseVoices[v].active)
        {
          slot = v;
          break;
        }
      }
      if (slot < 0)
        slot = (mReverseVoices[0].index >= mReverseVoices[1].index) ? 0 : 1;

      ReverseVoice& voice = mReverseVoices[slot];
      voice.active = true;
      voice.index = 0;
      voice.length = segmentFrames;
      voice.startReadPos = (mReverseWritePos + ringSize - 1) % ringSize;

      mReverseFramesUntilLaunch = std::max<size_t>(1, segmentFrames / 2);
    }

    // 2) Sum two windowed reversed taps -> wet; output = dry + wet * Mix; capture
    //    feeds back the wet sum (matches forward modes' use of post-tilt feedback).
    for (size_t c = 0; c < numChannels; c++)
    {
      const double inputSample = inputs[c][s];

      double wet = 0.0;
      for (int v = 0; v < 2; ++v)
      {
        const ReverseVoice& voice = mReverseVoices[v];
        if (!voice.active)
          continue;
        const size_t readPos = (voice.startReadPos + ringSize - voice.index) % ringSize;
        const double sample = mReverseRing[c][readPos];
        const double gain = _GetReverseWindowGain(voice.index, voice.length);
        wet += sample * gain;
      }

      // Tone tilt on wet only (matches Digital / Analog convention).
      wet = _ApplyToneTilt(c, wet, mTone, 3500.0);

      // Additive blend, identical to Digital / Analog. Pinned by
      // `Delay: Reverse and Digital RMS match within 0.5 dB at same Mix`.
      double finalSample = inputSample + wet * mMix;
      if (!std::isfinite(finalSample))
      {
        finalSample = 0.0;
        Reset();
      }

      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);
      mReverseRing[c][mReverseWritePos] = inputSample + wet * mFeedback;
    }

    // 3) Advance per-voice playback indices, write pointer, and stagger countdown.
    for (auto& voice : mReverseVoices)
    {
      if (!voice.active)
        continue;
      ++voice.index;
      if (voice.index >= voice.length)
        voice.active = false;
    }
    mReverseWritePos = (mReverseWritePos + 1) % ringSize;
    if (mReverseFramesUntilLaunch > 0)
      --mReverseFramesUntilLaunch;
  }

  return _GetPointers();
}

double Delay::_GetReverseWindowGain(size_t index, size_t length) const
{
  if (length < 2)
    return 0.0;
  if (index >= length)
    return 0.0;

  const double t = static_cast<double>(index) / static_cast<double>(length - 1);
  // Triangle (Age=0) and sin^2 (Age=1) both satisfy w(t) + w(t+0.5) = 1 at 50%
  // overlap, so the wet sum is constant regardless of Age. Triangle has a sharper
  // peak (more transient impact in the middle of each slice); sin^2 is smoother
  // ("Bloom").
  const double tri = (t < 0.5) ? 2.0 * t : 2.0 * (1.0 - t);
  const double sn = std::sin(M_PI * t);
  const double sinSq = sn * sn;
  return (1.0 - mAge) * tri + mAge * sinSq;
}

size_t Delay::_GetMaxFrames() const
{
  return std::max<size_t>(1, static_cast<size_t>(2.0 * mSampleRate));
}

} // namespace effect
} // namespace dsp
