import sounddevice as sd


class SoundRecorder:
    def __init__(self, samplerate=44100, default_duration=3, channels=1):
        """
        Init sounddevice library
        :param samplerate:
        :param duration:
        """
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        self._default_duration = default_duration

    def record(self, duration=None, blocking=True):
        """
        Simply record audio
        :param duration: duration of the recording
        :param blocking: if the recording is blocking or not
        :return:
        """
        if not duration:
            duration = self._default_duration
        return sd.rec(int(duration * sd.default.samplerate), blocking=blocking)

    def record_while(self, chunk_duration=None, limit_chunk=None):
        """
        Yield recorded audio chunks
        :param chunk_duration:
        :param limit_chunk:
        :return:
        """
        if not chunk_duration:
            chunk_duration = self._default_duration
        if not limit_chunk:
            while 1:
                yield self.record(chunk_duration)
        else:
            for i in range(limit_chunk):
                yield self.record(chunk_duration)

    def stream(self, callback, chunk_duration=None, total_duration=None):
        if not chunk_duration:
            chunk_duration = self._default_duration
        return sd.InputStream(callback=callback, blocksize=int(chunk_duration * sd.default.samplerate))
