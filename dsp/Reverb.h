//
//  Reverb.h
//  VoLum - Hall (FDN), Plate (Dattorro), Oktaverb, TremVerb
//
//  Iteration 2 (v0.9.0): added TremVerb mode and a per-mode sub-toggle that scales sub-mode
//  character (Hall: Studio/Concert/Cathedral, Plate: Steel/Brass/Copper, Oktaverb: Oct/Oct+5th/
//  Oct+SubOct, TremVerb: ignored). Oktaverb now produces octave-up (with optional fifth or
//  preserved sub-octave) feedback with a pitched pre-delay and per-line detune motion.
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
  enum Mode
  {
    kModeHall = 0,
    kModePlate = 1,
    kModeOktaverb = 2,
    kModeTremVerb = 3,
    kNumModes = 4,
  };

  Reverb();

  void Prepare(const size_t numChannels, const size_t numFrames, double sampleRate);

  // Legacy 7-arg API (backward compatibility - sub-mode defaults to 1, tremRate to 4 Hz).
  void SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode, double sampleRate);

  // Full iteration-2 API.
  // - subMode (0..2): per-mode meaning. Hall: Studio/Concert/Cathedral.
  //                   Plate: Steel/Brass/Copper. Oktaverb: Oct/Oct+5th/Oct+SubOct.
  //                   Ignored for TremVerb.
  // - tremRateHz (0.5..10): TremVerb tremolo rate.
  void SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode, double sampleRate,
                 int subMode, double tremRateHz);

  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  void _ProcessHall(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessOktaverb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessTremVerb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);

  void _AllocateHall();
  void _AllocatePlate();
  void _AllocateOktaverb();
  void _AllocatePreDelay();
  void _SetPreDelayLength(double preDelayMs);
  double _ReadWritePreDelay(double input);
  // Generic grain-based pitch shifter; ratio > 1 = pitch up, ratio < 1 = pitch down.
  // 'voice' indexes a separate set of grain buffers so shifters with different ratios don't
  // share state. We have kHallLines voices for octave-up, fifth and sub-octave each.
  enum PitchVoice { kVoiceOctUp = 0, kVoiceFifthUp = 1, kVoiceSubOct = 2, kNumPitchVoices = 3 };
  double _PitchShiftTick(int voice, int line, double input, double ratio);

  // Map raw mTone (0..10) to an LP cutoff for the active mode. Curve compresses the dark
  // side and keeps a usable bright top.
  double _ToneToCutoff(double minHz, double midHz, double maxHz) const;

  double mSampleRate = 0.0;
  double mMix = 0.3;
  double mDecay = 3.0;
  double mTone = 4.5;
  double mPreDelayMs = 20.0;
  double mShimmer = 0.5;
  int mMode = kModeHall;
  int mSubMode = 1; // 0..2 (per-mode meaning)
  double mTremRateHz = 4.0;

  // Hall (8-line FDN + Hadamard)
  static const int kHallLines = 8;
  std::vector<std::vector<double>> mHallDelays;
  std::vector<size_t> mHallIndices;
  std::vector<size_t> mHallLengths;
  std::vector<double> mHallLPState;
  double mHallLfoPhase = 0.0;
  // Per-line cathedral chorus state (Hall sub-mode 2 only): a slow detune LFO read tap.
  std::vector<double> mHallChorusPhase;

  // Pitch-shifter state per voice (0=Oct up, 1=Fifth up, 2=Sub-oct).
  // Each voice has kHallLines independent buffers.
  std::vector<std::vector<std::vector<double>>> mPitchBufs; // [voice][line][sample]
  std::vector<std::vector<size_t>> mPitchWriteIdx;
  std::vector<std::vector<double>> mPitchPhase;

  // Pitched pre-delay (Oktaverb): one buffer per line, one read tap.
  std::vector<std::vector<double>> mPitchedPreBuf;
  std::vector<size_t> mPitchedPreIdx;
  double mPitchedPreDelayMs = 60.0; // sub-mode-dependent

  // Per-line detune LFO for pitched line (Oktaverb motion).
  std::vector<double> mPitchedDetunePhase;

  // Plate (Dattorro)
  static const int kInputAPs = 4;
  std::vector<std::vector<double>> mInputAPBuf;
  std::vector<size_t> mInputAPIdx;
  std::vector<size_t> mInputAPLen;

  // Tank: 2 halves
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

  // Plate early-reflection short taps (densifies wet onset; sub-mode-dependent count).
  std::vector<double> mErTapMs;   // tap delays in ms
  std::vector<double> mErTapGain; // tap gains

  double mPlateLfoPhase = 0.0;

  // Shared pre-delay
  std::vector<double> mPreDelayBuf;
  size_t mPreDelayIdx = 0;
  size_t mPreDelayLen = 0;

  double mInputLPState = 0.0;

  // TremVerb LFO state (photocell-shaped wet-bus VCA).
  double mTremPhase = 0.0;

  bool mHallAllocated = false;
  bool mPlateAllocated = false;
  bool mOktaverbAllocated = false;
};

} // namespace effect
} // namespace dsp
