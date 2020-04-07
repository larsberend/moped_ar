import numpy as np


class CMad():
	def __init__(self, freq=400, gain=0.01):
		self.sampleFreq = np.double(freq) # sample frequency in Hz
		self.betaDef = np.double(gain)		      # 2 * proportional gain

		# ---------------------------------------------------------------------------------------------------
		# Variable definitions

		self.beta = self.betaDef							# 2 * proportional gain (Kp)
		self.quat = np.double((1,0,0,0))
		q0, q1, q2, q3 = self.quat	        # quaternion of sensor frame relative to auxiliary frame


	def MadgwickAHRSupdateIMU(self, gx, gy, gz, ax, ay, az):
		q0, q1, q2, q3 = self.quat
		beta = self.beta
		betaDef = self.betaDef
		sampleFreq = self.sampleFreq

		recipNorm = np.double(0)
		s0, s1, s2, s3 = np.double((0, 0, 0, 0))
		qDot1, qDot2, qDot3, qDot4 = np.double((0, 0, 0, 0))
		_2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3 = np.double((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

		# Rate of change of quaternion from gyroscope
		qDot1 = np.double(0.5) * (-q1 * gx - q2 * gy - q3 * gz)
		qDot2 = np.double(0.5) * (q0 * gx + q2 * gz - q3 * gy)
		qDot3 = np.double(0.5) * (q0 * gy - q1 * gz + q3 * gx)
		qDot4 = np.double(0.5) * (q0 * gz + q1 * gy - q2 * gx)

		# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
		if(not((np.isnan(ax)) and (np.isnan(ay)) and (np.isnan(az)))):

			# Normalise accelerometer measurement
			recipNorm = (ax * ax + ay * ay + az * az)**(np.double(-1/2))
			ax *= recipNorm
			ay *= recipNorm
			az *= recipNorm

			# Auxiliary variables to avoid repeated arithmetic
			_2q0 = 2.0 * q0
			_2q1 = 2.0 * q1
			_2q2 = 2.0 * q2
			_2q3 = 2.0 * q3
			_4q0 = 4.0 * q0
			_4q1 = 4.0 * q1
			_4q2 = 4.0 * q2
			_8q1 = 8.0 * q1
			_8q2 = 8.0 * q2
			q0q0 = q0 * q0
			q1q1 = q1 * q1
			q2q2 = q2 * q2
			q3q3 = q3 * q3

			# Gradient decent algorithm corrective step
			s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
			s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
			s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
			s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
			recipNorm = (s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)**(np.double(-1/2)) # normalise step magnitude
			s0 *= recipNorm
			s1 *= recipNorm
			s2 *= recipNorm
			s3 *= recipNorm

			# Apply feedback step
			qDot1 -= beta * s0
			qDot2 -= beta * s1
			qDot3 -= beta * s2
			qDot4 -= beta * s3


		# Integrate rate of change of quaternion to yield quaternion
		q0 += qDot1 * (1.0 / sampleFreq)
		q1 += qDot2 * (1.0 / sampleFreq)
		q2 += qDot3 * (1.0 / sampleFreq)
		q3 += qDot4 * (1.0 / sampleFreq)

		# Normalise quaternion
		recipNorm = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)**np.double(-1/2)
		q0 *= recipNorm
		q1 *= recipNorm
		q2 *= recipNorm
		q3 *= recipNorm
		# print((q0,q1,q2,q3))
		self.quat = q0, q1, q2, q3
		# print(self.quat)
		return q0, q1, q2, q3
