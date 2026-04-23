//
//  Reverb.h
//  VoLum
//

#pragma once

#include "dsp.h"
#include <vector>

namespace dsp
{
namespace effect
{

class Reverb : public DSP
{
public:
  Reverb();

  void SetParams(double mix, double decay, double tone, int mode, double sampleRate);
  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  double mSampleRate = 44100.0;
  double mMix = 0.5;
  double mDecay = 3.0;
  double mTone = 6.0;
  int mMode = 2; // 0=Spring, 1=Plate, 2=Hall, 3=Shimmer

  // Basic FDN structure (8 delay lines)
  static const int kNumLines = 8;
  std::vector<std::vector<double>> mDelayLines;
  std::vector<size_t> mDelayIndices;
  std::vector<size_t> mDelayLengths;
  std::vector<double> mDelayLowpass;

  // LFO for modulation
  double mLfoPhase = 0.0;
};

} // namespace effect
} // namespace dsp
