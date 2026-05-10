//
//  Delay.h
//  VoLum
//
//  Effect staging: mode order { Digital, Analog, Reverse }, with shared Tone/Age
//  controls and a global Ping-Pong toggle for the forward delay modes.
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
    kModeReverse = 2,
    kNumModes = 3,
  };

  Delay();

  void Prepare(const size_t numChannels, const size_t numFrames, double sampleRate);

  // Legacy 5-arg API (kept for backward compatibility with older call sites).
  // Defaults: tone=0.5, age=0.0, pingPong=false.
  void SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate);

  // Full iteration-2 API.
  // - tone (0..1): per-mode tilt EQ; 0.5 = flat.
  // - age (0..1): per-mode character control (Digital crusher/noise, Analog BBD darkness/chorus depth,
  //   Reverse fade-shape softness).
  // - pingPong: stereo cross-feedback toggle (R-line seed + opposite tap feedback). Ignored by Reverse.
  void SetParams(double timeMs, double feedback, double mix, int mode, double sampleRate,
                 double tone, double age, bool pingPong);

  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;
  void _PrepareDelayLines(const size_t numChannels);
  void _PrepareReverseBuffers(const size_t numChannels);
  void _ResetReverseState();

  DSP_SAMPLE** _ProcessDigital(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  DSP_SAMPLE** _ProcessAnalog(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  DSP_SAMPLE** _ProcessReverse(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  // Window gain for one playback voice. t = index/(length-1) in [0,1]. Blend of
  // triangular (Age=0) and sin^2 (Age=1) windows; both unit-sum at 50% overlap, so
  // two voices launched length/2 apart produce a constant-gain wet bus (no slice
  // boundary dip).
  double _GetReverseWindowGain(size_t index, size_t length) const;

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

  // Smoothing for time changes
  double mCurrentDelayFrames = 0.0;
  double mTargetDelayFrames = 0.0;

  // Ring buffers per channel
  std::vector<std::vector<double>> mBuffer;
  size_t mWriteIndex = 0;

  // Reverse mode: continuous capture ring + two overlap-add playback voices.
  // Each voice grabs a snapshot of the most-recent segmentFrames samples at launch
  // and plays them backwards; voices are staggered by length/2 so their windowed
  // sum is ~constant (no slice-boundary amplitude dip). In-flight voices keep
  // their original length when the time knob changes, so segment-length updates
  // do not glitch the wet bus.
  struct ReverseVoice
  {
    bool active = false;
    size_t index = 0;       // 0..length-1, position within reversed playback
    size_t length = 0;      // segmentFrames captured at launch
    size_t startReadPos = 0;// ring index of the most-recent sample at launch
  };
  std::vector<std::vector<double>> mReverseRing; // per-channel capture ring
  size_t mReverseRingSize = 0;
  size_t mReverseWritePos = 0;
  size_t mReverseSegmentFrames = 1;
  size_t mReverseFramesUntilLaunch = 1;
  ReverseVoice mReverseVoices[2];

  // Per-channel one-pole lowpass state for tone tilt + per-repeat HF damping.
  std::vector<double> mToneState;
  std::vector<double> mFeedbackLpState;

  // Analog optical chorus LFO.
  double mChorusPhase = 0.0;

  // Analog compander (peak follower for write-side compression / read-side expansion).
  double mCompandEnv = 0.0;
};

} // namespace effect
} // namespace dsp
