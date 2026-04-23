//
//  Reverb.cpp
//  VoLum
//

#include "Reverb.h"
#include <algorithm>
#include <cmath>

namespace dsp
{
namespace effect
{

static constexpr double kPI = 3.14159265358979323846;

Reverb::Reverb()
{
  mDelayLines.resize(kNumLines);
  mDelayIndices.resize(kNumLines, 0);
  mDelayLengths.resize(kNumLines, 0);
  mDelayLowpass.resize(kNumLines, 0.0);
}

void Reverb::SetParams(double mix, double decay, double tone, int mode, double sampleRate)
{
  if (mSampleRate != sampleRate)
  {
    mSampleRate = sampleRate;
    
    // Primes in ms scaled to frames
    const double baseLengthsMs[kNumLines] = { 31.0, 37.0, 43.0, 53.0, 61.0, 71.0, 79.0, 89.0 };
    for (int i = 0; i < kNumLines; i++) {
        mDelayLengths[i] = static_cast<size_t>((baseLengthsMs[i] / 1000.0) * mSampleRate) + 100; // +100 for modulation padding
        mDelayLines[i].assign(mDelayLengths[i] + 500, 0.0); // +500 for safety padding
        mDelayIndices[i] = 0;
    }
  }

  mMix = std::clamp(mix, 0.0, 1.0);
  mDecay = std::clamp(decay, 0.1, 10.0);
  mTone = std::clamp(tone, 0.0, 10.0);
  mMode = mode;
}

void Reverb::Reset()
{
  for (int i = 0; i < kNumLines; i++) {
      std::fill(mDelayLines[i].begin(), mDelayLines[i].end(), 0.0);
      mDelayIndices[i] = 0;
      mDelayLowpass[i] = 0.0;
  }
}

void Reverb::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  this->DSP::_PrepareBuffers(numChannels, numFrames);
}

DSP_SAMPLE** Reverb::Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  _PrepareBuffers(numChannels, numFrames);

  // RT60 to loop gain
  const double averageDelayTime = 58.0 / 1000.0; 
  double loopGain = std::pow(10.0, -3.0 * averageDelayTime / mDecay);
  if (loopGain >= 0.999) loopGain = 0.999;
  
  // Tone -> lowpass coefficient
  // 0 -> heavy damping, 10 -> bright
  double lpfCoef = 0.1 + (mTone / 10.0) * 0.8; 

  // Hadamard matrix for 8x8 mixing
  const double H = 0.35355339; // 1 / sqrt(8)
  double mixMatrix[8][8] = {
      { H,  H,  H,  H,  H,  H,  H,  H},
      { H, -H,  H, -H,  H, -H,  H, -H},
      { H,  H, -H, -H,  H,  H, -H, -H},
      { H, -H, -H,  H,  H, -H, -H,  H},
      { H,  H,  H,  H, -H, -H, -H, -H},
      { H, -H,  H, -H, -H,  H, -H,  H},
      { H,  H, -H, -H, -H, -H,  H,  H},
      { H, -H, -H,  H, -H,  H,  H, -H}
  };

  for (size_t s = 0; s < numFrames; s++)
  {
    mLfoPhase += 0.5 * 2.0 * kPI / mSampleRate; // 0.5 Hz LFO
    if (mLfoPhase > 2.0 * kPI) mLfoPhase -= 2.0 * kPI;

    // Downmix stereo input to mono for the FDN
    double inputSample = 0.0;
    for (size_t c = 0; c < numChannels; c++) {
      inputSample += inputs[c][s];
    }
    if (numChannels > 0) inputSample /= static_cast<double>(numChannels);

    // --- FDN Processing (Strictly ONCE per frame) ---
    double readVals[8] = {0};

    for (int i = 0; i < kNumLines; i++)
    {
      // Simple LFO modulation on the read length
      double mod = std::sin(mLfoPhase + (i * kPI / 4.0)) * 5.0; // 5 samples depth
      double readPos = static_cast<double>(mDelayIndices[i]) - static_cast<double>(mDelayLengths[i]) - mod;
      while (readPos < 0.0) readPos += mDelayLines[i].size();

      size_t idx1 = static_cast<size_t>(readPos);
      size_t idx2 = (idx1 + 1) % mDelayLines[i].size();
      double frac = readPos - idx1;

      double val = mDelayLines[i][idx1] * (1.0 - frac) + mDelayLines[i][idx2] * frac;

      // Apply 1-pole lowpass
      mDelayLowpass[i] = mDelayLowpass[i] * (1.0 - lpfCoef) + val * lpfCoef;
      readVals[i] = mDelayLowpass[i] * loopGain;
    }

    // Mix matrix feedback
    for (int i = 0; i < kNumLines; i++)
    {
      double feedback = 0.0;
      for (int j = 0; j < kNumLines; j++) {
          feedback += mixMatrix[i][j] * readVals[j];
      }
      
      mDelayLines[i][mDelayIndices[i]] = inputSample * 0.5 + feedback;
      mDelayIndices[i] = (mDelayIndices[i] + 1) % mDelayLines[i].size();
    }

    // --- Upmix and Output ---
    for (size_t c = 0; c < numChannels; c++)
    {
      double outSum = 0.0;
      for (int i = 0; i < kNumLines; i++)
      {
        // Accumulate to output (using alternating signs to spread stereo image)
        if (i % 2 == c % 2) {
             outSum += readVals[i];
        } else {
             outSum -= readVals[i];
        }
      }

      double finalSample = inputs[c][s] * (1.0 - mMix) + outSum * 0.3 * mMix;
      
      // NaN / Infinity protection to prevent ASIO driver hangs
      if (std::isnan(finalSample) || std::isinf(finalSample)) {
          finalSample = 0.0;
          Reset(); // Reset delay lines if it blew up
      }

      mOutputs[c][s] = static_cast<DSP_SAMPLE>(finalSample);
    }
  }

  return _GetPointers();
}

} // namespace effect
} // namespace dsp
