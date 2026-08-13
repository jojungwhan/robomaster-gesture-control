#ifndef LEAP_HAND_BRIDGE_H
#define LEAP_HAND_BRIDGE_H

#include <stdint.h>

#if defined(_WIN32)
#  define LHB_CALL __cdecl
#  if defined(LHB_BUILD_DLL)
#    define LHB_API __declspec(dllexport)
#  else
#    define LHB_API __declspec(dllimport)
#  endif
#else
#  define LHB_CALL
#  define LHB_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define LHB_MAX_HANDS 2

enum lhb_result {
  LHB_OK = 0,
  LHB_TIMEOUT = 1,
  LHB_ERROR_ARGUMENT = -1,
  LHB_ERROR_ALLOCATION = -2,
  LHB_ERROR_LEAP_CREATE = -3,
  LHB_ERROR_LEAP_OPEN = -4,
  LHB_ERROR_LEAP_POLL = -5
};

enum lhb_status_flag {
  LHB_STATUS_SERVICE_CONNECTED = 1,
  LHB_STATUS_DEVICE_PRESENT = 2
};

typedef struct lhb_hand_sample {
  uint32_t hand_id;
  int32_t hand_type;
  uint64_t visible_time_us;

  float palm_x;
  float palm_y;
  float palm_z;

  float velocity_x;
  float velocity_y;
  float velocity_z;

  float direction_x;
  float direction_y;
  float direction_z;

  float normal_x;
  float normal_y;
  float normal_z;

  float pinch_strength;
  float grab_strength;
  float pinch_distance_mm;
} lhb_hand_sample;

typedef struct lhb_frame_sample {
  uint64_t frame_id;
  int64_t sensor_timestamp_us;
  float framerate;
  uint32_t hand_count;
  uint32_t total_hand_count;
  uint32_t reserved;
  lhb_hand_sample hands[LHB_MAX_HANDS];
} lhb_frame_sample;

LHB_API int LHB_CALL lhb_create(void **out_context);
LHB_API int LHB_CALL lhb_poll(
    void *context,
    uint32_t timeout_ms,
    lhb_frame_sample *out_frame);
LHB_API uint32_t LHB_CALL lhb_status(void *context);
LHB_API uint32_t LHB_CALL lhb_frame_sample_size(void);
LHB_API uint32_t LHB_CALL lhb_hand_sample_size(void);
LHB_API uint32_t LHB_CALL lhb_last_leap_result(void *context);
LHB_API void LHB_CALL lhb_destroy(void *context);

#ifdef __cplusplus
}
#endif

#endif
