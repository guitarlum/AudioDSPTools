//
//  Reverb.h
//  VoLum - Hall (FDN), Plate (Dattorro), Oktaverb
//
//  Effect staging: Hall uses the good Cathedral-ish recipe under the single Hall label,
//  Plate is the original Dattorro plate, and Oktaverb exposes Dark / Shimmer / Bloom
//  sub-modes with pitch-in-feedback and bloom voicing.
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
    kNumModes = 3,
  };

  // Equal-power crossfade Mix law (replaces additive `dry + wet*mix`). Hall and Plate
  // wet busses have no internal wet-gain, so kReverbWetTrim trims them up so all three
  // reverb modes hit comparable perceived loudness at Mix=0.5. Oktaverb wet already
  // bakes sm.wetGain (1.40 / 1.55) into the wet bus, so it does NOT use this trim - its
  // internal 50 percent cap bounds the user-mix angle instead.
  static constexpr double kReverbWetTrim = 1.55;

  Reverb();

  void Prepare(const size_t numChannels, const size_t numFrames, double sampleRate);

  // Legacy 7-arg API (backward compatibility - sub-mode defaults to Oct for Oktaverb).
  void SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode, double sampleRate);

  // Staging API.
  // - subMode (0..2): Oktaverb only: Dark / Shimmer / Bloom.
  //                   Ignored by Hall and Plate.
  void SetParams(double mix, double decay, double tone, double preDelayMs, double shimmer, int mode, double sampleRate,
                 int subMode);

  void Reset();

  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

private:
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  void _ProcessHall(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessPlate(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);
  void _ProcessOktaverb(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames);

  // Input diffusion shared by the FDN modes, and the early taps that give them wet
  // energy before the shortest delay line can produce any.
  double _DiffuseFdnInput(double input);
  void _FdnEarlyField(double diffused, double& outL, double& outR);

  void _AllocateHall();
  void _AllocatePlate();
  void _AllocateOktaverb();
  void _AllocatePreDelay();
  void _SetPreDelayLength(double preDelayMs);
  double _ReadWritePreDelay(double input);
  // Generic grain-based pitch shifter; ratio > 1 = pitch up, ratio < 1 = pitch down.
  // 'voice' indexes separate grain buffers so feedback octave-up, feedback octave-down, and
  // parallel sub-fifth paths never share state.
  enum PitchVoice
  {
    kVoiceOctUpFeedback = 0,
    kVoiceOctDownFeedback = 1,
    kVoiceSubFifthParallel = 2,
    kNumPitchVoices = 3
  };
  double _PitchShiftTick(int voice, int line, double input, double ratio);

  // Map raw mTone (0..10) to an LP cutoff for the active mode. Curve compresses the dark
  // side and keeps a usable bright top.
  double _ToneToCutoff(double minHz, double midHz, double maxHz) const;

  double mSampleRate = 0.0;
  double mMix = 0.3;
  // One-pole smoothed Mix used by the final dry/wet crossfade so automating the
  // Mix knob (in a DAW or via a long preset crossfade) does not zipper at block
  // boundaries. SetParams updates mMix (target); the process loops advance
  // mMixSmoothed toward it per sample. Reset snaps them together.
  double mMixSmoothed = 0.3;
  double mMixSmoothCoef = 0.0;
  double mDecay = 3.0;
  double mTone = 4.5;
  double mPreDelayMs = 20.0;
  double mShimmer = 0.5;
  int mMode = kModeHall;
  int mSubMode = 0; // Oktaverb only: 0=Dark, 1=Shimmer, 2=Bloom.

  // Hall (8-line FDN + Hadamard)
  static const int kHallLines = 8;
  std::vector<std::vector<double>> mHallDelays;
  std::vector<size_t> mHallIndices;
  std::vector<size_t> mHallLengths;
  std::vector<double> mHallLPState;
  double mHallLfoPhase = 0.0;
  // Per-line cathedral chorus state (Hall sub-mode 2 only): a slow detune LFO read tap.
  std::vector<double> mHallChorusPhase;

  // Nested allpass diffuser in front of the FDN (Hall and Oktaverb). Without it the
  // tank is excited by a bare impulse, so the first pass through the eight lines came
  // out as eight separate clicks instead of a dense burst. Plate keeps its own
  // Dattorro input diffusers; these are separate buffers so the two never share state.
  static const int kFdnDiffusers = 4;
  std::vector<std::vector<double>> mFdnAPBuf;
  std::vector<size_t> mFdnAPIdx;
  std::vector<size_t> mFdnAPLen;

  // The early field, which is the diffuser output taken through one more allpass per
  // channel. Without it the wet bus is silent until the shortest line (52.7 ms on
  // Hall), which both left an audible hole and put a hidden floor under PRE-DLY.
  //
  // An allpass has flat magnitude, so this adds energy without colouring anything. The
  // first thing tried here was a set of fixed taps read from inside the delay lines,
  // and that turned out to be eight delayed copies of one signal summed together - a
  // comb filter, measuring 12 dB of level swing between 220 Hz and 382 Hz. Density
  // comes from the shared chain in front; these two only decorrelate left from right.
  static const int kFdnEarlyAPs = 2;
  std::vector<std::vector<double>> mFdnEarlyAPBuf;
  std::vector<size_t> mFdnEarlyAPIdx;
  std::vector<size_t> mFdnEarlyAPLen;

  // Pitch-shifter state per voice (0=Oct up, 1=Fifth up, 2=Sub-oct).
  // Each voice has kHallLines independent buffers.
  std::vector<std::vector<std::vector<double>>> mPitchBufs; // [voice][line][sample]
  std::vector<std::vector<size_t>> mPitchWriteIdx;
  std::vector<std::vector<double>> mPitchPhase;

  std::vector<double> mOktaverbLfoPhase;
  std::vector<double> mOktaverbPitchLPState;
  std::vector<double> mOktaverbFeedbackPitchLPState;
  std::vector<double> mOktaverbFeedbackPitchHPState;
  double mBloomEnv = 0.0;
  double mBloomVCA = 0.0;

  // Plate (Dattorro)
  static const int kInputAPs = 4;
  std::vector<std::vector<double>> mInputAPBuf;
  std::vector<size_t> mInputAPIdx;
  std::vector<size_t> mInputAPLen;

  // Tank: 2 halves. Each half is decay-diffusion-1 -> delay -> damping ->
  // decay-diffusion-2 -> delay, per Dattorro. Through 1.2.0 only the first allpass and
  // first delay existed, which dropped the very stages that make the two halves
  // different lengths: 672 + 4453 and 908 + 4217 both come to 5125 samples, so both
  // halves rang in lockstep and produced one centred echo every 172 ms.
  struct TankHalf
  {
    std::vector<double> apBuf;
    size_t apIdx = 0;
    size_t apLen = 0;
    std::vector<double> delBuf;
    size_t delIdx = 0;
    size_t delLen = 0;
    double lpState = 0.0;
    std::vector<double> ap2Buf;
    size_t ap2Idx = 0;
    size_t ap2Len = 0;
    std::vector<double> del2Buf;
    size_t del2Idx = 0;
    size_t del2Len = 0;
    double lastOut = 0.0;
  };
  TankHalf mTank[2];

  // Dattorro's output network: seven taps per channel, each reading inside a tank
  // buffer belonging mostly to the opposite half. This is what fills the time between
  // loop passes; taking a single tap at the end of each half, as 1.2.0 did, meant the
  // loop period itself was the thing you heard.
  // Offsets only; which buffer each one reads is fixed and spelled out in _ProcessPlate.
  static const int kPlateOutTaps = 7;
  size_t mPlateTapL[kPlateOutTaps] = {0};
  size_t mPlateTapR[kPlateOutTaps] = {0};

  double mPlateLfoPhase = 0.0;

  // Shared pre-delay
  std::vector<double> mPreDelayBuf;
  size_t mPreDelayIdx = 0;
  size_t mPreDelayLen = 0;

  double mInputLPState = 0.0;

  bool mHallAllocated = false;
  bool mPlateAllocated = false;
  bool mOktaverbAllocated = false;
};

} // namespace effect
} // namespace dsp
