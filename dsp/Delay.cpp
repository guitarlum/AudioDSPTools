//
//  Delay.cpp
//  VoLum
//

#include "Delay.h"
#include <algorithm>
#include <cmath>

namespace dsp
{
namespace effect
{

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
  mMode = std::clamp(mode, 0, 3);

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
  {
    std::fill(buf.begin(), buf.end(), 0.0);
  }
  mWriteIndex = 0;
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

DSP_SAMPLE** Delay::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

  if (mMode == 3)
    return _ProcessReverse(inputs, numChannels, numFrames);

  const size_t maxFrames = _GetMaxFrames();
  
  for (size_t s = 0; s < numFrames; s++)
  {
    // Smooth delay time to avoid clicks on change
    mCurrentDelayFrames += 0.001 * (mTargetDelayFrames - mCurrentDelayFrames);
    
    for (size_t c = 0; c < numChannels; c++)
    {
      double inputSample = inputs[c][s];
      double delayFrames = mCurrentDelayFrames;
      
      // Ping pong logic (alternating channels)
      if (mMode == 2 && numChannels > 1) {
          if (c == 1) {
              delayFrames *= 2.0; // Right channel delayed twice as much
          }
      }

      // Fractional delay read (linear interpolation)
      double readPos = static_cast<double>(mWriteIndex) - delayFrames;
      if (readPos < 0.0) readPos += maxFrames;
      
      size_t idx1 = static_cast<size_t>(readPos);
      size_t idx2 = (idx1 + 1) % maxFrames;
      double frac = readPos - idx1;
      
      double delayedSample = mBuffer[c][idx1] * (1.0 - frac) + mBuffer[c][idx2] * frac;
      
      // Tape mode lowpass simple approximation
      if (mMode == 0) {
          // Very crude high frequency roll-off by just slightly damping the delayed sample
          delayedSample *= 0.95;
      }

      // Parallel mix: dry at unity, wet added on top (like a send/return)
      double finalSample = inputSample + delayedSample * mMix;

      // NaN / Infinity protection
      if (std::isnan(finalSample) || std::isinf(finalSample)) {
          finalSample = 0.0;
          Reset();
      }

      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);
      
      // Write to buffer with feedback
      mBuffer[c][mWriteIndex] = inputSample + delayedSample * mFeedback;
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
      const double reversedSample = mReversePlaybackReady ? mReversePlaybackBuffer[c][playbackIndex] * fadeGain : 0.0;

      double finalSample = inputSample * (1.0 - mMix) + reversedSample * mMix;

      if (std::isnan(finalSample) || std::isinf(finalSample)) {
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
  if (fadeFrames == 0)
    return 1.0;

  const size_t endDistance = segmentFrames - 1 - index;
  const size_t fadeDistance = std::min(index, endDistance);
  if (fadeDistance >= fadeFrames)
    return 1.0;

  return static_cast<double>(fadeDistance) / static_cast<double>(fadeFrames);
}

size_t Delay::_GetMaxFrames() const
{
  return std::max<size_t>(1, static_cast<size_t>(2.0 * mSampleRate));
}

} // namespace effect
} // namespace dsp
