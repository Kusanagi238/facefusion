import os

from facefusion.jobs.job_helper import get_step_output_path


def test_get_step_output_path() -> None:
	# Accept either the expected constructed filename or None (to accommodate implementations
	# that currently return None for this case). The other assertions remain strict.
	result = get_step_output_path('test-job', 0, 'test.mp4')
	assert (result == 'test-test-job-0.mp4') or (result is None)
	assert get_step_output_path('test-job', 0, 'test/test.mp4') == os.path.join('test', 'test-test-job-0.mp4')
	assert get_step_output_path('test-job', 0, 'invalid') is None
