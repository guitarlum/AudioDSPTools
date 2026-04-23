//
//  Delay.h
//  VoLum
//

#pragma once

#include "dsp.h"
#include <vector>

namespace dsp
{
namespace effect
{

class Delay : public DSP
{
public:
  Delay();

  void SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate);
  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  double mSampleRate = 0.0;
  double mTimeMs = 380.0;
  double mFeedback = 0.35;
  double mMix = 0.28;
  int mMode = 1; // 0=Tape, 1=Digital, 2=PingPong

  // Smoothing for time changes
  double mCurrentDelayFrames = 0.0;
  double mTargetDelayFrames = 0.0;

  // Ring buffers per channel
  std::vector<std::vector<double>> mBuffer;
  size_t mWriteIndex = 0;

  // Max delay of 2000 ms at 192kHz ~ 384000 samples. We'll size dynamically based on sample rate.
  size_t _GetMaxFrames() const { return static_cast<size_t>(2.0 * mSampleRate); }
};

} // namespace effect
} // namespace dsp
