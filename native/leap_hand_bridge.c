#include "leap_hand_bridge.h"

#include "LeapC.h"

#include <Windows.h>
#include <stdlib.h>
#include <string.h>

typedef struct lhb_context {
  LEAP_CONNECTION connection;
  uint32_t status;
  eLeapRS last_result;
} lhb_context;

static void copy_hand(const LEAP_HAND *source, lhb_hand_sample *destination) {
  memset(destination, 0, sizeof(*destination));
  destination->hand_id = source->id;
  destination->hand_type = (int32_t)source->type;
  destination->visible_time_us = source->visible_time;

  /*
   * Gemini 6.2 can report a valid hand while leaving stabilized_position at
   * (0, 0, 0) for the original Leap Motion Controller.  The SDK samples use
   * palm.position, which remains populated and is the appropriate input for
   * our relative virtual joystick; smoothing is applied in Python.
   */
  destination->palm_x = source->palm.position.x;
  destination->palm_y = source->palm.position.y;
  destination->palm_z = source->palm.position.z;

  destination->velocity_x = source->palm.velocity.x;
  destination->velocity_y = source->palm.velocity.y;
  destination->velocity_z = source->palm.velocity.z;

  destination->direction_x = source->palm.direction.x;
  destination->direction_y = source->palm.direction.y;
  destination->direction_z = source->palm.direction.z;

  destination->normal_x = source->palm.normal.x;
  destination->normal_y = source->palm.normal.y;
  destination->normal_z = source->palm.normal.z;

  destination->pinch_strength = source->pinch_strength;
  destination->grab_strength = source->grab_strength;
  destination->pinch_distance_mm = source->pinch_distance;
}

int LHB_CALL lhb_create(void **out_context) {
  lhb_context *context;

  if (!out_context) {
    return LHB_ERROR_ARGUMENT;
  }
  *out_context = NULL;

  context = (lhb_context *)calloc(1, sizeof(*context));
  if (!context) {
    return LHB_ERROR_ALLOCATION;
  }

  context->last_result = LeapCreateConnection(NULL, &context->connection);
  if (context->last_result != eLeapRS_Success) {
    free(context);
    return LHB_ERROR_LEAP_CREATE;
  }

  context->last_result = LeapOpenConnection(context->connection);
  if (context->last_result != eLeapRS_Success) {
    LeapDestroyConnection(context->connection);
    free(context);
    return LHB_ERROR_LEAP_OPEN;
  }

  *out_context = context;
  return LHB_OK;
}

int LHB_CALL lhb_poll(
    void *opaque_context,
    uint32_t timeout_ms,
    lhb_frame_sample *out_frame) {
  lhb_context *context = (lhb_context *)opaque_context;
  ULONGLONG deadline;

  if (!context || !out_frame) {
    return LHB_ERROR_ARGUMENT;
  }

  memset(out_frame, 0, sizeof(*out_frame));
  deadline = GetTickCount64() + (ULONGLONG)timeout_ms;

  for (;;) {
    LEAP_CONNECTION_MESSAGE message;
    uint32_t remaining = 0;

    memset(&message, 0, sizeof(message));
    if (timeout_ms > 0) {
      ULONGLONG now = GetTickCount64();
      if (now >= deadline) {
        return LHB_TIMEOUT;
      }
      remaining = (uint32_t)(deadline - now);
    }

    context->last_result =
        LeapPollConnection(context->connection, remaining, &message);
    if (context->last_result == eLeapRS_Timeout) {
      return LHB_TIMEOUT;
    }
    if (context->last_result != eLeapRS_Success) {
      return LHB_ERROR_LEAP_POLL;
    }

    if (message.type == eLeapEventType_Connection) {
      context->status |= LHB_STATUS_SERVICE_CONNECTED;
    } else if (message.type == eLeapEventType_ConnectionLost) {
      context->status = 0;
    } else if (message.type == eLeapEventType_Device) {
      context->status |= LHB_STATUS_DEVICE_PRESENT;
    } else if (message.type == eLeapEventType_DeviceLost ||
               message.type == eLeapEventType_DeviceFailure) {
      context->status &= ~LHB_STATUS_DEVICE_PRESENT;
    } else if (message.type == eLeapEventType_Tracking &&
               message.tracking_event) {
      const LEAP_TRACKING_EVENT *tracking = message.tracking_event;
      uint32_t count = tracking->nHands;
      uint32_t index;

      if (count > LHB_MAX_HANDS) {
        count = LHB_MAX_HANDS;
      }

      out_frame->frame_id = (uint64_t)tracking->info.frame_id;
      out_frame->sensor_timestamp_us = tracking->info.timestamp;
      out_frame->framerate = tracking->framerate;
      out_frame->hand_count = count;
      out_frame->total_hand_count = tracking->nHands;

      for (index = 0; index < count; ++index) {
        copy_hand(&tracking->pHands[index], &out_frame->hands[index]);
      }
      return LHB_OK;
    }

    if (timeout_ms == 0) {
      return LHB_TIMEOUT;
    }
  }
}

uint32_t LHB_CALL lhb_status(void *opaque_context) {
  lhb_context *context = (lhb_context *)opaque_context;
  return context ? context->status : 0;
}

uint32_t LHB_CALL lhb_frame_sample_size(void) {
  return (uint32_t)sizeof(lhb_frame_sample);
}

uint32_t LHB_CALL lhb_hand_sample_size(void) {
  return (uint32_t)sizeof(lhb_hand_sample);
}

uint32_t LHB_CALL lhb_last_leap_result(void *opaque_context) {
  lhb_context *context = (lhb_context *)opaque_context;
  return context ? (uint32_t)context->last_result : 0;
}

void LHB_CALL lhb_destroy(void *opaque_context) {
  lhb_context *context = (lhb_context *)opaque_context;
  if (!context) {
    return;
  }
  if (context->connection) {
    LeapCloseConnection(context->connection);
    LeapDestroyConnection(context->connection);
  }
  free(context);
}
