//
//  Delay.h
//  VoLum
//
//  Iteration 2 (v0.9.0): mode order { Digital, Analog, Tape, Reverse }, with shared
//  Tone/Age controls and a global Ping-Pong toggle.
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
  enum Mode
  {
    kModeDigital = 0,
    kModeAnalog = 1,
    kModeTape = 2,
    kModeReverse = 3,
    kNumModes = 4,
  };

  enum TapeSubMode
  {
    kTapeStudio = 0,
    kTapeVintage = 1,
    kTapeBroken = 2,
  };

  Delay();

  void Prepare(const size_t numChannels, const size_t numFrames, double sampleRate);

  // Legacy 5-arg API (kept for backward compatibility with older call sites).
  // Defaults: tone=0.5, age=0.0, pingPong=false, tapeSubMode=Vintage.
  void SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate);

  // Full iteration-2 API.
  // - tone (0..1): per-mode tilt EQ; 0.5 = flat.
  // - age (0..1): per-mode character control (Tape wow/sat, Analog BBD darkness/chorus depth,
  //   Digital crusher/noise, Reverse fade-shape softness).
  // - pingPong: stereo cross-feedback toggle. Ignored by Reverse.
  // - tapeSubMode (0..2): Studio/Vintage/Broken; only used when mode == Tape.
  void SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate,
                 double tone, double age, bool pingPong, int tapeSubMode);

  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;
  void _PrepareDelayLines(const size_t numChannels);
  void _PrepareReverseBuffers(const size_t numChannels);
  void _ResetReverseState();

  DSP_SAMPLE** _ProcessDigital(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  DSP_SAMPLE** _ProcessAnalog(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  DSP_SAMPLE** _ProcessTape(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  DSP_SAMPLE** _ProcessReverse(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  double _GetReverseFadeGain(size_t index, size_t segmentFrames) const;

  // Per-mode tone-tilt one-pole filter applied on the wet bus.
  // Tone in [0,1]: 0=darker, 0.5=flat, 1=brighter. Returns filtered sample.
  double _ApplyToneTilt(size_t channel, double sample, double tone, double cutoffHz);

  size_t _GetMaxFrames() const;

  double mSampleRate = 0.0;
  double mTimeMs = 380.0;
  double mFeedback = 0.35;
  double mMix = 0.28;
  int mMode = kModeDigital;
  double mTone = 0.5;
  double mAge = 0.0;
  bool mPingPong = false;
  int mTapeSubMode = kTapeVintage;

  // Smoothing for time changes
  double mCurrentDelayFrames = 0.0;
  double mTargetDelayFrames = 0.0;

  // Ring buffers per channel
  std::vector<std::vector<double>> mBuffer;
  size_t mWriteIndex = 0;

  // Reverse mode uses one buffer to capture the next slice while another plays back the previous slice backwards.
  std::vector<std::vector<double>> mReverseCaptureBuffer;
  std::vector<std::vector<double>> mReversePlaybackBuffer;
  size_t mReverseSegmentFrames = 1;
  size_t mReverseIndex = 0;
  bool mReversePlaybackReady = false;

  // Per-channel one-pole lowpass state for tone tilt + per-repeat HF damping.
  std::vector<double> mToneState;
  std::vector<double> mFeedbackLpState;

  // Tape wow/flutter LFO accumulators (single global pair, per-channel phase offset applied at read).
  double mWowPhase = 0.0;
  double mFlutterPhase = 0.0;

  // Analog optical chorus LFO.
  double mChorusPhase = 0.0;

  // Analog compander (peak follower for write-side compression / read-side expansion).
  double mCompandEnv = 0.0;
};

} // namespace effect
} // namespace dsp
