import numpy as np
class CMad(freq = 400, gain=0.1):
    self.sampleFreq = np.double(freq) # sample frequency in Hz
    self.betaDef = np.double(gain)		      # 2 * proportional gain




    # ---------------------------------------------------------------------------------------------------
    # Variable definitions

    self.beta = betaDef								# 2 * proportional gain (Kp)
    self.q0, self.q1, self.q2, self.q3 = np.double((1,0,0,0))	        # quaternion of sensor frame relative to auxiliary frame
    print('ok')

    def MadgwickAHRSupdateIMU(gx, gy, gz, ax, ay, az):
    	recipNorm = np.double(0)
    	s0, s1, s2, s3 = np.double((0, 0, 0, 0))
    	qDot1, qDot2, qDot3, qDot4 = np.double((0, 0, 0, 0))
    	_2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3 = np.double((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    	# Rate of change of quaternion from gyroscope
    	qDot1 = np.double(0.5) * (-self.q1 * gx - self.q2 * gy - self.q3 * gz)
    	qDot2 = np.double(0.5) * (self.q0 * gx + self.q2 * gz - self.q3 * gy)
    	qDot3 = np.double(0.5) * (self.q0 * gy - self.q1 * gz + self.q3 * gx)
    	qDot4 = np.double(0.5) * (self.q0 * gz + self.q1 * gy - self.q2 * gx)

    	# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    	if(not((ax == 0.0) and (ay == 0.0) and (az == 0.0))):

    		# Normalise accelerometer measurement
    		recipNorm = (ax * ax + ay * ay + az * az)**(np.double(-1/2))
    		ax *= recipNorm
    		ay *= recipNorm
    		az *= recipNorm

    		# Auxiliary variables to avoid repeated arithmetic
    		_2q0 = 2.0 * self.q0
    		_2q1 = 2.0 * self.q1
    		_2q2 = 2.0 * self.q2
    		_2q3 = 2.0 * self.q3
    		_4q0 = 4.0 * self.q0
    		_4q1 = 4.0 * self.q1
    		_4q2 = 4.0 * self.q2
    		_8q1 = 8.0 * self.q1
    		_8q2 = 8.0 * self.q2
    		q0q0 = self.q0 * self.q0
    		q1q1 = self.q1 * self.q1
    		q2q2 = self.q2 * self.q2
    		q3q3 = self.q3 * self.q3

    		# Gradient decent algorithm corrective step
    		s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
    		s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * self.q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
    		s2 = 4.0 * q0q0 * self.q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
    		s3 = 4.0 * q1q1 * self.q3 - _2q1 * ax + 4.0 * q2q2 * self.q3 - _2q2 * ay
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
    	self.q0 += qDot1 * (1.0 / sampleFreq)
    	self.q1 += qDot2 * (1.0 / sampleFreq)
    	self.q2 += qDot3 * (1.0 / sampleFreq)
    	self.q3 += qDot4 * (1.0 / sampleFreq)

    	# Normalise quaternion
    	recipNorm = (self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)**np.double(-1/2)
    	self.q0 *= recipNorm
    	self.q1 *= recipNorm
    	self.q2 *= recipNorm
    	self.q3 *= recipNorm
    	return self.q0, self.q1, self.q2, self.q3
