import numpy as np

m_to_m = 10
m_to_tube = 60
m_height = 30
tube_length = 100
sum = m_to_m + m_to_tube

slit_size = 1.6
focal_length = 1.9

x_max = (m_height * sum - tube_length * m_to_m) / (tube_length - m_height)
print(x_max)

aov = 2 * np.arctan(slit_size / (2 * focal_length))
nom = 2 * np.tan(aov / 2)
d1 = tube_length / nom
d2 = m_height / nom
print(d1)
print(d2)
