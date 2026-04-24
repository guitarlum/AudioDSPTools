//
//  Reverb.h
//  VoLum — Hall (FDN) + Plate (Dattorro) reverb
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

  // mode: 0=Hall, 1=Plate
  void SetParams(double mix, double decay, double tone, int mode, double sampleRate);
  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  void _ProcessHall(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);

  void _AllocateHall();
  void _AllocatePlate();

  double mSampleRate = 0.0;
  double mMix = 0.5;
  double mDecay = 3.0;
  double mTone = 6.0;
  int mMode = 0; // 0=Hall, 1=Plate

  // ─── Hall (8-line FDN + Hadamard) ────────────────────────────
  static const int kHallLines = 8;
  std::vector<std::vector<double>> mHallDelays;
  std::vector<size_t> mHallIndices;
  std::vector<size_t> mHallLengths;
  std::vector<double> mHallLPState;
  double mHallLfoPhase = 0.0;

  // ─── Plate (Dattorro) ────────────────────────────────────────
  // Input diffusion: 4 allpass filters
  static const int kInputAPs = 4;
  std::vector<std::vector<double>> mInputAPBuf;
  std::vector<size_t> mInputAPIdx;
  std::vector<size_t> mInputAPLen;

  // Tank: 2 halves, each has: decay-diffusor AP, delay, damping LP
  struct TankHalf {
    std::vector<double> apBuf;
    size_t apIdx = 0;
    size_t apLen = 0;

    std::vector<double> delBuf;
    size_t delIdx = 0;
    size_t delLen = 0;

    double lpState = 0.0;
    double lastOut = 0.0;
  };
  TankHalf mTank[2];

  // Modulation
  double mPlateLfoPhase = 0.0;

  // Pre-delay
  std::vector<double> mPreDelayBuf;
  size_t mPreDelayIdx = 0;
  size_t mPreDelayLen = 0;

  // Input lowpass
  double mInputLPState = 0.0;

  bool mHallAllocated = false;
  bool mPlateAllocated = false;
};

} // namespace effect
} // namespace dsp
