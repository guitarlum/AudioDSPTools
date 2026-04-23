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

void Delay::SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate)
{
  if (mSampleRate != sampleRate)
  {
    mSampleRate = sampleRate;
    for (auto& buf : mBuffer)
    {
      buf.assign(_GetMaxFrames(), 0.0);
    }
    mWriteIndex = 0;
  }

  mTimeMs = std::clamp(timeMs, 10.0, 2000.0);
  mFeedback = std::clamp(feedback, 0.0, 0.99);
  mMix = std::clamp(mix, 0.0, 1.0);
  mMode = mode;

  mTargetDelayFrames = (mTimeMs / 1000.0) * mSampleRate;
  if (mCurrentDelayFrames == 0.0)
    mCurrentDelayFrames = mTargetDelayFrames;
}

void Delay::Reset()
{
  for (auto& buf : mBuffer)
  {
    std::fill(buf.begin(), buf.end(), 0.0);
  }
  mWriteIndex = 0;
}

void Delay::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  this->DSP::_PrepareBuffers(numChannels, numFrames);
  
  if (mBuffer.size() != numChannels)
  {
    mBuffer.resize(numChannels, std::vector<double>(_GetMaxFrames(), 0.0));
  }
}

DSP_SAMPLE** Delay::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

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

      // Output mix
      double finalSample = inputSample * (1.0 - mMix) + delayedSample * mMix;

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

} // namespace effect
} // namespace dsp
