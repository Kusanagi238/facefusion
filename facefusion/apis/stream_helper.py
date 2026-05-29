import ctypes
import os
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import cv2
import numpy
from starlette.websockets import WebSocket

from facefusion import face_store, rtc, rtc_store, state_manager, streamer
from facefusion.audio import create_empty_audio_frame
from facefusion.codecs import aom_decoder, aom_encoder, opus_decoder, opus_encoder, vpx_decoder, vpx_encoder
from facefusion.libraries import datachannel as datachannel_module
from facefusion.types import AomDecoder, AomEncoder, AudioCodec, AudioFrame, PeerConnection, Resolution, RtcPeer, SdpAnswer, SdpOffer, SessionId, VideoCodec, VisionFrame, VpxDecoder, VpxEncoder


async def process_image(websocket : WebSocket) -> None:
	source_paths = state_manager.get_item('source_paths')

	if source_paths:
		capture_vision_frame = await anext(receive_vision_frames(websocket), None)

		if numpy.any(capture_vision_frame):
			output_vision_frame = streamer.process_frame(create_empty_audio_frame(), capture_vision_frame)
			is_success, output_frame_buffer = cv2.imencode('.jpg', output_vision_frame)

			if is_success:
				await websocket.send_bytes(output_frame_buffer.tobytes())


#TODO: needs review
def process_video(session_id : SessionId, sdp_offer : SdpOffer) -> Optional[SdpAnswer]:
	video_codec : VideoCodec = 'vp8'

	if rtc.get_payload_type(sdp_offer, 'av1'):
		video_codec = 'av1'

	video_payload_type = rtc.get_payload_type(sdp_offer, video_codec)

	if video_payload_type:
		peer_connection : PeerConnection = rtc.create_peer_connection()
		video_receiver_track = rtc.add_video_track(peer_connection, 'recvonly', video_codec, video_payload_type)
		video_sender_track = rtc.add_video_track(peer_connection, 'sendonly', video_codec, video_payload_type)

		audio_codec : AudioCodec = 'opus'
		audio_payload_type = rtc.get_payload_type(sdp_offer, audio_codec)
		audio_receiver_track = None
		audio_sender_track = None

		if audio_payload_type:
			audio_receiver_track = rtc.add_audio_track(peer_connection, 'recvonly', audio_codec, audio_payload_type)
			audio_sender_track = rtc.add_audio_track(peer_connection, 'sendonly', audio_codec, audio_payload_type)

		rtc.set_remote_description(peer_connection, sdp_offer)
		local_sdp = rtc.create_sdp_answer(peer_connection)

		if local_sdp:
			rtc_peer : RtcPeer =\
			{
				'peer_connection': peer_connection,
				'video':
				{
					'sender_track': video_sender_track,
					'receiver_track': video_receiver_track,
					'codec': video_codec
				}
			}

			if audio_receiver_track and audio_sender_track:
				rtc_peer['audio'] =\
				{
					'sender_track': audio_sender_track,
					'receiver_track': audio_receiver_track,
					'codec': audio_codec
				}

			rtc_store.init_peers(session_id)
			rtc_store.get_peers(session_id).append(rtc_peer)

			threading.Thread(target = run_peer_loop, args = (session_id, rtc_peer), daemon = True).start()
			return local_sdp

		datachannel_module.create_static_library().rtcDeletePeerConnection(peer_connection)

	return None


async def receive_vision_frames(websocket : WebSocket) -> AsyncIterator[VisionFrame]:
	websocket_event = await websocket.receive()

	while websocket_event.get('type') == 'websocket.receive':
		frame_buffer = websocket_event.get('bytes') or bytes()
		vision_frame = cv2.imdecode(numpy.frombuffer(frame_buffer, numpy.uint8), cv2.IMREAD_COLOR)

		if numpy.any(vision_frame):
			yield vision_frame

		websocket_event = await websocket.receive()


def swap_status_label(swap_status : bool) -> str:
	if swap_status:
		return 'YES'
	return 'NO'


def swap_status_failed(swap_status : bool) -> bool:
	if swap_status:
		return False
	return True


def draw_debug_overlay(vision_frame : VisionFrame, incoming_fps : float, inference_fps : float, outgoing_fps : float, inference_duration : float, encode_duration : float, wait_duration : float, queue_size : int, frame_index : int, resolution : Resolution, swap_status : bool) -> None:
	metrics : List[Tuple[str, str, bool]] =\
	[
		('IN', str(int(incoming_fps)) + ' FPS', incoming_fps < 15),
		('INF', str(int(inference_fps)) + ' FPS', inference_fps < 15),
		('OUT', str(int(outgoing_fps)) + ' FPS', outgoing_fps < 15),
		('INF TIME', str(int(inference_duration * 1000)) + ' ms', inference_duration > 0.05),
		('ENC TIME', str(int(encode_duration * 1000)) + ' ms', encode_duration > 0.01),
		('WAIT', str(int(wait_duration * 1000)) + ' ms', wait_duration > 0.05),
		('QUEUE', str(queue_size), queue_size == 0),
		('SWAP', swap_status_label(swap_status), swap_status_failed(swap_status)),
		('FRAME', str(frame_index), False),
		('RES', str(resolution[0]) + 'x' + str(resolution[1]), False)
	]

	for index, metric in enumerate(metrics):
		label, value, is_bad = metric
		color = (0, 255, 0)

		if is_bad:
			color = (0, 0, 255)

		y_position = 30 + index * 30
		cv2.putText(vision_frame, label + ' ' + value, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def encode_and_send(video_codec : VideoCodec, video_encoder : VisionFrame, output_vision_buffer : bytes, resolution : Resolution, frame_index : int, rtc_peer : RtcPeer, audio_encoder : VisionFrame, audio_frame : AudioFrame, duration_out : list) -> None:
	encode_start = time.monotonic()
	output_video_buffer = encode_video_frame(video_codec, video_encoder, output_vision_buffer, resolution, frame_index)
	duration_out[0] = time.monotonic() - encode_start
	send_timestamp = time.monotonic()

	if output_video_buffer:
		rtc.send_video(rtc_peer, output_video_buffer, int(send_timestamp * 90000))

	if audio_encoder and audio_frame.dtype == numpy.float32:
		output_audio_buffer = opus_encoder.encode(audio_encoder, audio_frame.tobytes(), 960)

		if output_audio_buffer:
			rtc.send_audio(rtc_peer, output_audio_buffer, int(send_timestamp * 48000))


#TODO: needs review
def run_peer_loop(session_id : SessionId, rtc_peer : RtcPeer) -> None:
	frame_deque : Deque[Tuple[VisionFrame, AudioFrame]] = deque(maxlen = 4)
	frame_event : threading.Event = threading.Event()
	latest_audio_frame : List[AudioFrame] = [create_empty_audio_frame()]
	receiver_threads : List[threading.Thread] = []

	video_codec : VideoCodec = rtc_peer.get('video').get('codec')
	video_track : int = rtc_peer.get('video').get('receiver_track')
	incoming_fps_value : List[float] = [0.0]
	video_receiver_thread = threading.Thread(target = receive_video_frames, args = (video_track, video_codec, frame_deque, incoming_fps_value, latest_audio_frame, frame_event), daemon = True)
	receiver_threads.append(video_receiver_thread)

	if rtc_peer.get('audio'):
		audio_codec : AudioCodec = 'opus'
		audio_track = rtc_peer.get('audio').get('receiver_track')
		audio_receiver_thread = threading.Thread(target = receive_audio_frames, args = (audio_track, audio_codec, latest_audio_frame), daemon = True)
		receiver_threads.append(audio_receiver_thread)

	for receiver_thread in receiver_threads:
		receiver_thread.start()

	frame_event.wait()
	frame_event.clear()
	temp_vision_frame, audio_frame = frame_deque.popleft()

	if numpy.any(temp_vision_frame):
		temp_resolution : Resolution = (temp_vision_frame.shape[1], temp_vision_frame.shape[0])
		video_encoder = create_video_encoder(video_codec, temp_resolution)
		audio_encoder = opus_encoder.create(48000, 2)
		frame_index = 0
		inference_fps = 0.0
		outgoing_fps = 0.0
		encode_duration = 0.0
		wait_duration = 0.0
		loop_start = time.monotonic()
		encode_thread : Optional[threading.Thread] = None
		encode_thread_duration : List[float] = [0.0]
		is_debug : bool = state_manager.get_item('log_level') == 'debug'
		log_file : Optional[object] = None
		reference_faces : List[bool] = []
		stream_start : float = time.monotonic()
		video_fps : float = 23.976

		if is_debug:
			os.makedirs('.logs', exist_ok = True)
			log_path = os.path.join('.logs', 'stream-helper-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
			log_file = open(log_path, 'w')
			reference_path = 'reference-bella.log'

			if os.path.isfile(reference_path):
				with open(reference_path) as ref_file:
					for ref_line in ref_file:
						reference_faces.append(ref_line.startswith('face:1'))

		prev_frame_hash : int = 0
		duplicate_count : int = 0

		while numpy.any(temp_vision_frame):
			if frame_index > 0 and frame_index % 20 == 0:
				face_store.clear_static_faces()
			inference_start = time.monotonic()
			output_vision_frame = streamer.process_frame(audio_frame, temp_vision_frame)
			inference_duration = time.monotonic() - inference_start
			swap_status = False

			if is_debug:
				current_frame_hash : int = hash(temp_vision_frame.tobytes())

				if current_frame_hash == prev_frame_hash:
					duplicate_count += 1

				prev_frame_hash = current_frame_hash
				swap_status = numpy.any(output_vision_frame != temp_vision_frame)

			if inference_duration > 0:
				inference_fps = 1.0 / inference_duration

			if encode_thread:
				encode_thread.join()
				encode_duration = encode_thread_duration[0]

			loop_duration = time.monotonic() - loop_start

			if loop_duration > 0:
				outgoing_fps = 1.0 / loop_duration

			loop_start = time.monotonic()

			if is_debug:
				elapsed = time.monotonic() - stream_start
				ref_index = min(int(elapsed * video_fps), len(reference_faces) - 1)
				ref_face = False

				if reference_faces and ref_index >= 0:
					ref_face = reference_faces[ref_index]

				missed = ref_face and swap_status_failed(swap_status)
				draw_debug_overlay(output_vision_frame, incoming_fps_value[0], inference_fps, outgoing_fps, inference_duration, encode_duration, wait_duration, len(frame_deque), frame_index, temp_resolution, swap_status)
				log_file.write('[' + datetime.now().strftime('%H:%M:%S.%f')[:12] + '] in:' + str(int(incoming_fps_value[0])) + ' inf:' + str(int(inference_fps)) + ' out:' + str(int(outgoing_fps)) + ' inf_ms:' + str(int(inference_duration * 1000)) + ' enc_ms:' + str(int(encode_duration * 1000)) + ' wait_ms:' + str(int(wait_duration * 1000)) + ' fq:' + str(len(frame_deque)) + ' dup:' + str(duplicate_count) + ' swap:' + str(int(swap_status)) + ' ref:' + str(ref_index) + ' ref_face:' + str(int(ref_face)) + ' missed:' + str(int(missed)) + ' frame:' + str(frame_index) + ' res:' + str(temp_resolution[0]) + 'x' + str(temp_resolution[1]) + '\n')
			output_resolution : Resolution = (output_vision_frame.shape[1], output_vision_frame.shape[0])
			output_vision_buffer = cv2.cvtColor(output_vision_frame, cv2.COLOR_BGR2YUV_I420).tobytes()

			if output_resolution != temp_resolution:
				destroy_video_encoder(video_codec, video_encoder)
				temp_resolution = output_resolution
				video_encoder = create_video_encoder(video_codec, temp_resolution)
				frame_index = 0

			encode_thread_duration = [0.0]
			encode_thread = threading.Thread(target = encode_and_send, args = (video_codec, video_encoder, output_vision_buffer, temp_resolution, frame_index, rtc_peer, audio_encoder, audio_frame, encode_thread_duration), daemon = True)
			encode_thread.start()

			frame_index += 1
			wait_start = time.monotonic()

			if frame_deque:
				frame_event.set()

			frame_event.wait()
			frame_event.clear()

			wait_duration = time.monotonic() - wait_start
			temp_vision_frame, audio_frame = frame_deque.popleft()

		if encode_thread:
			encode_thread.join()

		if log_file:
			log_file.write('duplicates:' + str(duplicate_count) + ' total:' + str(frame_index) + '\n')
			log_file.close()
		destroy_video_encoder(video_codec, video_encoder)  # TODO: remove unconditional destroy methods, which have no impact on control flow
		opus_encoder.destroy(audio_encoder)

	for receiver_thread in receiver_threads:
		receiver_thread.join()

	rtc_store.delete_peers(session_id)


def receive_video_frames(video_track : int, video_codec : VideoCodec, frame_deque : Deque[Tuple[VisionFrame, AudioFrame]], incoming_fps_value : list, latest_audio_frame : list, frame_event : threading.Event) -> None:
	datachannel_library = datachannel_module.create_static_library()
	video_decoder = create_video_decoder(video_codec)
	receive_buffer = ctypes.create_string_buffer(512 * 1024)
	receive_status_code = -3
	receive_start = time.monotonic()
	target_interval = 1.0 / 30

	while receive_status_code == 0 or receive_status_code == -3:
		buffer_size = ctypes.c_int(512 * 1024)
		receive_status_code = datachannel_library.rtcReceiveMessage(video_track, receive_buffer, ctypes.byref(buffer_size))

		if receive_status_code == 0 and buffer_size.value > 0:
			frame_buffer = receive_buffer.raw[:buffer_size.value]
			vision_frame = decode_video_frame(video_codec, video_decoder, frame_buffer)

			if numpy.any(vision_frame):
				current_time = time.monotonic()

				if current_time - receive_start >= target_interval:
					receive_duration = current_time - receive_start

					if receive_duration > 0:
						incoming_fps_value[0] = 1.0 / receive_duration

					receive_start = current_time
					frame_deque.append((vision_frame, latest_audio_frame[0]))
					frame_event.set()

		if receive_status_code == -3:
			time.sleep(0.001)

	frame_deque.append((numpy.empty(0), create_empty_audio_frame()))
	frame_event.set()
	destroy_video_decoder(video_codec, video_decoder)


def receive_audio_frames(audio_track : int, audio_codec : AudioCodec, latest_audio_frame : list) -> None:
	datachannel_library = datachannel_module.create_static_library()
	audio_decoder = opus_decoder.create(48000, 2)
	receive_buffer = ctypes.create_string_buffer(8 * 1024)
	receive_status_code = -3

	while receive_status_code == 0 or receive_status_code == -3:
		buffer_size = ctypes.c_int(8 * 1024)
		receive_status_code = datachannel_library.rtcReceiveMessage(audio_track, receive_buffer, ctypes.byref(buffer_size))

		if receive_status_code == 0 and buffer_size.value > 0:
			opus_buffer = receive_buffer.raw[:buffer_size.value]
			output_buffer = opus_decoder.decode(audio_decoder, opus_buffer, 960, 2)

			if output_buffer:
				latest_audio_frame[0] = numpy.frombuffer(output_buffer, dtype = numpy.float32)

		if receive_status_code == -3:
			time.sleep(0.001)

	opus_decoder.destroy(audio_decoder)


def decode_video_frame(video_codec : VideoCodec, video_decoder : VpxDecoder | AomDecoder, frame_buffer : bytes) -> Optional[VisionFrame]:
	if video_codec == 'av1':
		aom_pointer = aom_decoder.decode(video_decoder, frame_buffer)

		if aom_pointer:
			frame_width, frame_height = aom_pointer.get('resolution')
			vision_frame = numpy.frombuffer(aom_pointer.get('buffer'), dtype = numpy.uint8).reshape((frame_height * 3 // 2, frame_width))
			return cv2.cvtColor(vision_frame, cv2.COLOR_YUV2BGR_I420)

	if video_codec == 'vp8':
		vpx_pointer = vpx_decoder.decode(video_decoder, frame_buffer)

		if vpx_pointer:
			frame_width, frame_height = vpx_pointer.get('resolution')
			vision_frame = numpy.frombuffer(vpx_pointer.get('buffer'), dtype = numpy.uint8).reshape((frame_height * 3 // 2, frame_width))
			return cv2.cvtColor(vision_frame, cv2.COLOR_YUV2BGR_I420)

	return None


def encode_video_frame(video_codec : VideoCodec, video_encoder : VpxEncoder | AomEncoder, raw_frame_bytes : bytes, resolution : Resolution, frame_index : int) -> bytes:
	if video_codec == 'av1':
		return aom_encoder.encode(video_encoder, raw_frame_bytes, resolution, frame_index)

	if video_codec == 'vp8':
		return vpx_encoder.encode(video_encoder, raw_frame_bytes, resolution, frame_index)

	return bytes()


def create_video_decoder(video_codec : VideoCodec) -> Optional[VpxDecoder | AomDecoder]:
	if video_codec == 'av1':
		return aom_decoder.create(8)

	if video_codec == 'vp8':
		return vpx_decoder.create(8)

	return None


def create_video_encoder(video_codec : VideoCodec, resolution : Resolution) -> Optional[VpxEncoder | AomEncoder]:
	if video_codec == 'av1':
		return aom_encoder.create(resolution, 8000, 8, 10)

	if video_codec == 'vp8':
		return vpx_encoder.create(resolution, 8000, 8, 10)

	return None


def destroy_video_decoder(video_codec : VideoCodec, video_decoder : VpxDecoder | AomDecoder) -> None:
	if video_codec == 'av1':
		aom_decoder.destroy(video_decoder)

	if video_codec == 'vp8':
		vpx_decoder.destroy(video_decoder)


def destroy_video_encoder(video_codec : VideoCodec, video_encoder : VpxEncoder | AomEncoder) -> None:
	if video_codec == 'av1':
		aom_encoder.destroy(video_encoder)

	if video_codec == 'vp8':
		vpx_encoder.destroy(video_encoder)


def destroy_stream(session_id : SessionId) -> bool:
	if rtc_store.has_peers(session_id):
		rtc_store.delete_peers(session_id)
		return not rtc_store.has_peers(session_id)

	return False
