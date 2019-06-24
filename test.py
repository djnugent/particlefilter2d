
import cv2
import numpy as np

pix_per_unit = 20
image_size = (500,700,3)
sensor_range = 30


num_particle = 1000
state_vector_size = (num_particle,4) # x+y+heading+probability
# Create 1000 particles uniformly sampled over state space (35x35 grid) with the same probability
particles = np.random.uniform((0,0,0,1),(35,25,np.pi,1),state_vector_size)

actual_robot_pos = np.array((5,5,0))

def world_to_image(pnt):
    xi = int(pnt[0] * pix_per_unit)
    yi = int(image_size[0] - (pnt[1] * pix_per_unit))
    return xi,yi

def draw_robot(image,robot_pos):
    x0,y0,heading = robot_pos
    x0i,y0i = world_to_image((x0,y0))
    color = (255,34,4)
    cv2.circle(image, (x0i,y0i), 6,color,1,-1)
    x1 = sensor_range * np.cos(heading) + x0
    y1 = sensor_range * np.sin(heading) + y0
    x1i,y1i = world_to_image((x1,y1))
    cv2.line(image, (x0i,y0i), (x1i,y1i), color, 1)
    return image

def draw_walls(image,wall_map):
    lineThickness = 2
    for seg in wall_map:
        x0,y0 = seg.pt0
        x1,y1 = seg.pt1
        x0 *= pix_per_unit
        y0 = image_size[0] - (y0 * pix_per_unit)
        x1 *= pix_per_unit
        y1 = image_size[0] - (y1 * pix_per_unit)
        cv2.line(image, (x0,y0), (x1,y1), (0,255,0), lineThickness)
    return image

def draw_particles(image,particles):
    for particle in particles:
        x,y,heading,prob = particle
        x = int(x * pix_per_unit)
        y = int(image_size[0] - (y * pix_per_unit))
        color = int(prob * 255)
        cv2.circle(image, (x,y), 1,color,1,-1)
    return image



class Segment():

    def __init__(self,pt0,pt1):
        self.pt0 = pt0
        self.pt1 = pt1

    def intersection(other):
        pt_a0x,  pt_a0y = self.pt0
        pt_a1x,  pt_a1y = self.pt1
        pt_b0x,  pt_b0y = other.pt0
        pt_b1x,  pt_b1y = other.pt1
        s1_x = pt_a1x - pt_a0x
        s1_y = pt_a1y - pt_a0y
        s2_x = pt_b1x - pt_b0x
        s2_y = pt_b1y - pt_b0y

        s = (-s1_y * (pt_a0x - pt_b0x) + s1_x * (pt_a0y - pt_b0y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = ( s2_x * (pt_a0y - pt_b0y) - s2_y * (pt_a0x - pt_b0x)) / (-s2_x * s1_y + s1_x * s2_y)

        if s >= 0 and s <= 1 and t >= 0 and t <= 1:
            # Collision detected
            i_x = pt_a0x + (t * s1_x)
            i_y = pt_a0y + (t * s1_y)

        return None
wall_map = [
    Segment((1,4),(1,8)),
    Segment((1,8),(5,21)),
    Segment((5,21),(8,21)),
    Segment((8,21),(12,18)),
    Segment((12,18),(20,18)),
    Segment((20,18),(20,16)),
    Segment((20,16),(25,13)),
    Segment((25,13),(25,8)),
    Segment((25,8),(30,8)),
    Segment((30,8),(30,3)),
    Segment((30,3),(10,3)),
    Segment((10,3),(10,4)),
    Segment((10,4),(1,4))
]

image = np.ones(image_size,dtype=np.uint8) * 255
img = draw_walls(image,wall_map)
img = draw_particles(image,particles)
img = draw_robot(image,actual_robot_pos)
cv2.imshow("map",img)
cv2.waitKey(0)
