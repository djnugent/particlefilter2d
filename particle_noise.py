import cv2
import numpy as np

num_particle = 100

def draw(particles):
    image = np.zeros((255,255),dtype=np.uint8)
    for p in particles:
        y,x,head = np.array(p,dtype=np.uint8)
        cv2.circle(image,(x,y),1,255,1)
    cv2.imshow("image",image)
    cv2.waitKey(0)


# Create 1000 particles at pos 50,50 and heading 0
part = np.random.uniform((50,50,0),(50,50,0),(num_particle,3))

# Define gaussian noise for movement
bias = np.array((0,0),dtype=np.float) # (odom_bias, heading_bias)
noise = np.array((0.1,np.pi/90),dtype=np.float) # (odom_noise, heading_noise)

# Apply movement 200 times
# Move forward but dont turn
step = np.array((1,0),dtype=np.float)
for i in range(0,200):

    movement = np.random.normal(step + bias,noise,(num_particle,2))
    part[:,2] += movement[:,1]
    part[:,0] += movement[:,0] * np.sin(part[:,2])
    part[:,1] += movement[:,0] * np.cos(part[:,2])

    draw(part)

