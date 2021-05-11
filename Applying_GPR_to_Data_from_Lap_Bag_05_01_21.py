import csv
import numpy as np
import math

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# GET NECESSARY LAP DATA FROM STATE-SPACE MODEL (will use for 'test' data):
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

# Transforms individual state data matrix (i.e. data for x-position) from lap bag from global reference frame
# to one relative to the car. (4/7/2021)

# Transform a point's-worth of (x,y) data from the global perspective to the car's POV via rotation.
# NOTE:
# 1. x and y are vectors of x- and y-position data from the point prior @ time t
# 2. psi is a vector of change in theta data btwn points t and t+1 
def rotateXY(x, y, psi):
    psi = psi * -1
    xy_t = np.zeros((x.shape[0], 2))
    xy_t_plus_1 = np.zeros((x.shape[0], 2))
    xy_t[:,0] = x[:,0]
    xy_t[:,1] = y[:,0]
    # x- and y- position needs to be rotated, as axes have rotated when looking
    # at coordinates relative to car rather than global origin. Do so via the rotation
    # transformation matrix R = [cosT -sinT; sinT cosT]
    for i in range(len(xy_t)):
        rotation_transformation_matrix = np.array([[math.cos(psi[i]), -1 * math.sin(psi[i])],
                                                   [math.sin(psi[i]), math.cos(psi[i])]])
        # print(xy_initial[i])
        xy_t_plus_1[i] = rotation_transformation_matrix @ xy_t[i]
        # print(xy_final[i])
    return xy_t_plus_1 # np.subtract(xy_t_plus_1, xy_t)

# Takes the difference btwn each theta point to generate the changes in a state in the global POV
# to then serve as the respective set of state values relative to the car's POV
def get_Change_in_State(data_matrix):
    new_data_matrix = np.zeros((data_matrix.shape[0] - 1, data_matrix.shape[1]))
    for i in range(len(data_matrix) - 1):
        new_data_matrix[i] = data_matrix[i+1] - data_matrix[i]
    return new_data_matrix

with open("Rosbag_Lap_Data_03_29_2021.csv", 'r', newline = '') as optimalPathDataFile:
    # Create reader object for ros_bag with 2235 points of lap data:
    
    reader = csv.reader(optimalPathDataFile, delimiter = ',') # Might need to swap delimiter to comma
    # fields = line, time_stamp, time, speed_x, speed_y, speed_z, position_x, position_y, position_z, ...
    # ... orientation_x, orientation_y, orientation_z (header angle), break pedal, steering angle, steering wheel speed

    # Skip non-data containing rows in file
    next(reader)
    # for row in reader:
    #     print(row[3])
    
    
    # Define arrays to store states and inputs of kinematic bicycle model from rosbag lap data:
    pts_in_file = 2235
    x_data = np.zeros([pts_in_file, 1])             # Note: x-position in units of CHECK
    y_data = np.zeros([pts_in_file, 1])             # Note: y-position in units of CHECK
    theta_data = np.zeros([pts_in_file, 1])         # Note: Heading angle in units of [rads]
    v_data = np.zeros([pts_in_file, 1])             # Note: x-position in units of CHECK
    steerAngle_data = np.zeros([pts_in_file, 1])    # Note: Steering angle in units of [rads]
    
    states = np.zeros([pts_in_file, 3])
    linearized_states_t_plus_1 = np.zeros([pts_in_file - 1, 3])         # Had to edit this and below line to accomodate for making data relative to car (4/7/2021)
    inputs = np.zeros([pts_in_file - 1, 2])
    
    index = 0
    for row in reader:
        # Get state data:
        x_data[index] = float(row[6])           # Encapsulate all data in float() b/c each is read as a string
        y_data[index] = float(row[7])
        theta_data[index] = float(row[11])
        
        # Get input data:
        v_data[index] = math.sqrt(float(row[3])**2 + float(row[4])**2 + float(row[5])**2) # v = sqrt(v_x^2 + v_y^2 + v_z^2) CHECK IF NEED Z-DATA (think so)
        steerAngle_data[index] = float(row[13])
        index += 1
    
    #  (4/7/2021)
    # Make state info relative to car's frame of reference (rosbag returns info wrt global reference frame, if unedited can lead to unforeseen error)
    # x_data = makeRelativeToCar(x_data)
    # y_data = makeRelativeToCar(y_data)
        
    # Truncate first point of x and y-position data
    x_data_deltas = get_Change_in_State(x_data)
    y_data_deltas = get_Change_in_State(y_data)
    xy_deltas = np.zeros([pts_in_file - 1, 2]) # Not used until end of code
    
    # Get theta values relative to car's reference frame
    theta_data_relative = get_Change_in_State(theta_data)
    
    # Get x- and y-position values relative to car's reference frame
    xy_data_relative = rotateXY(x_data_deltas, y_data_deltas, theta_data_relative)
    # print(xy_data_relative)
    
    states[:,0] = x_data[:,0]
    states[:,1] = y_data[:,0]
    states[:,2] = theta_data[:,0]
    states_t_plus_1 = states[1:]
    
    # Consolidate all state info into single matrix  (4/7/2021)
    linearized_states_t_plus_1[:,0] = xy_data_relative[:,0]
    linearized_states_t_plus_1[:,1] = xy_data_relative[:,1]
    linearized_states_t_plus_1[:,2] = theta_data_relative[:,0]
    # NOTE: x_lt+1 = linearized_states_t_plus_1
    
    # print(linearized_states_t_plus_1)
    
    # Create inputs matrix, truncating the first point of inputs
    v_data_truncated = v_data[1:]
    steerAngle_data_truncated = steerAngle_data[1:]
    inputs[:,0] = v_data_truncated[:,0]
    inputs[:,1] = steerAngle_data_truncated[:,0]
    
    # print('Inital states: ' + str(linearized_states_t_plus_1[0]))
    # print('Min v: ' + str(np.amin(v_data)))
    # print('Max v: ' + str(np.amax(v_data)))
    # print('Min delta: ' + str(np.amin(steerAngle_data)))
    # print('Max delta: ' + str(np.amax(steerAngle_data)))
    
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# GENERATE TRAINING DATA FOR GPR:
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# arbitrary values
x_0 = -300.3330186            # Enter initial x-value from lap bag (4/7/2021)
y_0 = -1841.360897            # Enter initial x-value from lap bag (4/7/2021)
theta_0 = -1.553335116        # Enter initial x-value from lap bag [radians] (4/7/2021)
L = 10      # NEEDS UPDATED!!!
dt = 1.0    # Okay to leave at 1.0?
v_0 = 27            # Set to value little below minimum value of velocity in lap bag
delta_0 = -0.375    # Set to value little below minimum value of steering angle in lap bag [radians]
numberOfStates = 3
numberOfInputs = 2

# Re-defined KBM to process all state info to predict next step for each (x,y,theta) pt in imported lap data
class KBM:
    def __init__(self, v, delta, theta):
        # Define changing rate for each output of model using given input values
        self.v = v
        self.delta = delta
        self.x_dot = self.v * np.cos(self.delta + theta)
        self.y_dot = self.v * np.sin(self.delta + theta)
        self.theta_dot = self.v * np.sin(self.delta) / L
        
    def getnextstate(self, x, y, theta, t):
        # Update output values using a discrete time model
        update_vector = np.zeros((x.shape[0], numberOfStates))
        # print(update_vector[:,0][:, np.newaxis].shape)
        update_vector[:,0][:, np.newaxis] = x + self.x_dot * t
        update_vector[:,1][:, np.newaxis] = y + self.y_dot * t
        update_vector[:,2][:, np.newaxis] = theta + self.theta_dot * t
        return update_vector

# Get states predicted by KBM
# NOTE: p_t+1 = states_predicted
bicycle_model = KBM(v_data, steerAngle_data, theta_data)
states_predicted = bicycle_model.getnextstate(x_data, y_data, theta_data, dt)

# Isolate predicted (x,y) states 
x_predicted = np.zeros([(states_predicted[:,0]).shape[0] - 1, 1])
y_predicted = np.zeros([(states_predicted[:,0]).shape[0] - 1, 1])
x_predicted_deltas = get_Change_in_State(states_predicted[:,0][:, np.newaxis])
y_predicted_deltas = get_Change_in_State(states_predicted[:,1][:, np.newaxis])
# print(x_predicted)
# print()
# print(y_predicted)

# Make predicted theta state relative to car's frame of reference
theta_predicted_relative = get_Change_in_State(states_predicted[:,2][:, np.newaxis])

# Get predicted (x,y) states relative to car's frame of reference
xy_predicted_relative = rotateXY(x_predicted_deltas, y_predicted_deltas, theta_predicted_relative)

# Consolidate all predicted state info into single matrix (4/13/2021)
# NOTE: p_lt+1 = linearized_states_predicted_t
linearized_states_predicted_t = np.zeros((xy_predicted_relative[:,0].shape[0], 3))
linearized_states_predicted_t[:,0] = xy_predicted_relative[:,0]
linearized_states_predicted_t[:,1] = xy_predicted_relative[:,1]
linearized_states_predicted_t[:,2] = theta_predicted_relative[:,0]
# print(linearized_states_predicted_t)


"""
Old code from pre-4/13/2021; used for generation of predicted data:

class KBM:
    def __init__(self, v, delta):
        # Define changing rate for each output of model using given input values
        self.v = v
        self.delta = delta
        self.x_dot = self.v * math.cos(self.delta + theta_0)
        self.y_dot = self.v * math.sin(self.delta + theta_0)
        self.theta_dot = self.v * math.sin(self.delta) / L
        
    def getnextstate(self, t):
        # Update output values using a discrete time model
        update_vector = np.zeros(3)
        update_vector[0] = x_0 + self.x_dot * t
        update_vector[1] = y_0 + self.y_dot * t
        update_vector[2] = theta_0 + self.theta_dot * t
        return update_vector
    

class DataSet:
    def __init__(self, size_i, size_j, step_i, step_j):
        # Instantiate model and 3d array
        self.outputs = KBM(v_0, delta_0)
        self.nextstate = [x_0, y_0, theta_0]
        self.size_i = size_i
        self.size_j = size_j
        self.step_i = step_i
        self.step_j = step_j
        self.inputs = np.zeros((size_i*size_j, numberOfInputs))      # Having trouble making this size dynamic; needs tinkering each execution
        # print(self.inputs.shape) 
        self.set = np.zeros([int(self.size_i), int(self.size_j), 3])
        
        
    def update(self, v, delta):
        # Update to next position
        self.outputs = KBM(v, delta)
        self.nextstate = self.outputs.getnextstate(dt)
        
    def buildset(self):
        # Fill dataset values
        for i in range(0, self.size_i, 1):
            for j in range(0, self.size_j, 1):
                self.update((v_0+(i*self.step_i)), (delta_0+(np.deg2rad(j*self.step_j))))
                # print('% .2f % .2f' %((v_0+(i*self.step_i)), (delta_0+(np.deg2rad(j*self.step_j)))))
                self.inputs[self.size_i*i+j] = np.array([v_0+(i*self.step_i), delta_0+(np.deg2rad(j*self.step_j))])
                # print(self.nextstate)
                # print()
                self.set[i, j, 0] = self.nextstate[0]
                self.set[i, j, 1] = self.nextstate[1]
                self.set[i, j, 2] = self.nextstate[2]
        # print(self.inputs) 
        
# Main:
n_dimension = 20
m_dimension = 20
test = DataSet(n_dimension, m_dimension, 0.7, 2.5)  # Produces a (n*m,#inputs) inputs matrix, and a (n,m,#states) outputs matrix
                                            # The inputs matrix was chosen such that it goes from a value a little below
                                            # each input's min. value to a little above their respective max value.
test.update(300, 0)
# print(test.nextstate)
test.buildset()

states_predicted = np.reshape(test.set, ((n_dimension * m_dimension, numberOfStates)))
inputs_for_predictons = test.inputs

# print((states_predicted[:,1])) # print predicted (x,y) of point 0.
# print(states_predicted[:][1])
# print()
# print(inputs_for_predictons)
XY_predicted = np.zeros([(states_predicted[:,0][:, np.newaxis]).shape[0] - 1, 2])
print(XY_predicted.shape)
print(XY_predicted[:,0].shape)
XY_predicted[:,0][:, np.newaxis] = get_Change_in_State(states_predicted[:,0][:, np.newaxis])
XY_predicted[:,1][:, np.newaxis] = get_Change_in_State(states_predicted[:,1][:, np.newaxis])
print(XY_predicted)
print(states_predicted)
"""


    
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# USE GPR AND KINEMATIC BICYCLE MODEL TO ESTIMATE ERROR IN LQR CONTROLLER LAP DATA:
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def Kernel(X, Y, L = 1):
    """
    Generates a kernel matrix from 
    X = array of shape [N, D] where N is the number of points, D is the dimensionality of each point
    Y = array of shape [M, D] where M is the number of poitns in the array, same dimensionality for each point
    """
    def k_func(x, y):
        """
        kernel function 
        """
        return np.exp(-(1/(2*(L**2)))*(np.linalg.norm(x - y)))
    XC = X.shape[0]
    YC = Y.shape[0]
    kernel = np.zeros((XC, YC))
    for i in range(XC):
        for j in range(YC):
            kernel[i,j] = k_func(X[i], Y[j])
    return kernel

# Get X_test and Y_test
# X_test = inputs
# Y_test = states

# Everything below this need polishing.
# Main observation: Should the state and input data be our test data and not our training data?
# And for training data, should we get that via the kinematic bicycle model?

"""
# Get X_test and Y_test (smaller set of sample inputs and smaller respective dataset respectively)
numberOfInputsPerPt = 2
numberOfStates = 3   # x, y, theta
testSetPts = 2
trainingInputsCol_1 = np.zeros(testSetPts)[:, np.newaxis]
trainingInputsCol_2 = np.zeros(testSetPts)[:, np.newaxis]
X_test = np.zeros((testSetPts**2, numberOfInputsPerPt))  
Y_test = np.zeros((testSetPts**2, numberOfStates))  

# Select random test 
indexSet1 = np.random.choice((v_data.shape[0]), size = testSetPts, replace=False)
indexSet2 = np.random.choice((steerAngle_data.shape[0]), size = testSetPts, replace=False)

counter = 0
for i in range(testSetPts):
    for j in range(testSetPts):
        # print(v_data[indexSet1[i]])
        # print(steerAngle_data[indexSet2[j]])
        # print(X_test[counter,: ].shape)
        
        temp_array = np.array([v_data[indexSet1[i]], steerAngle_data[indexSet2[j]]])
        temp_array = temp_array.reshape((1,2))
        X_test[counter] = temp_array        # Stopped tinkering here: 3/29/2021
        # print(X_test.shape)
        
        # index = 10*indexSet1[i] + indexSet2[j]
        # Y_test[counter] += Y_train[index]
        counter += 1
"""

# Training data is the state and input data linearized wrt the car @ time = t+1
X_train = inputs
# Y_train = (x_lt+1) + (error_lt+1) = (x_lt+1) + [(x_lt+1) - (p_lt+1)]
Y_train = linearized_states_t_plus_1 + (linearized_states_t_plus_1 - linearized_states_predicted_t)
print(Y_train)

# Test data is the state and input data linearized wrt the car @ time = t (states all = 0 as a result)
X_test = inputs
Y_test = np.zeros((inputs.shape[0], 3)) # Y_test = x_lp = zeros-vector
# Y_test = states_predicted[1:]
print()
print(Y_test)

print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_test shape: ' + str(Y_test.shape))
# print(linearized_states_predicted_t)
# print()
# print(linearized_states_t_plus_1 - linearized_states_predicted_t)


# print()
# # print(linearized_states_t_plus_1)
# print(states_t_plus_1[0:].shape)
# print()

# print()
# print(states_predicted[1:].shape)
# print()


# Get the required kernels
Kss = Kernel(Y_train, Y_train)
# Kst = Kernel(Y_train, Y_test)
Kts = Kernel(Y_test, Y_train) # This equals Kst.T
# Ktt = Kernel(Y_test, Y_test)


u = np.ones((Y_train.shape[1])) * 0. # I made it 0 for simplicity... but other values work
nc = 0.01
fs = Kts @ np.linalg.inv(Kss + nc * np.eye(X_train.shape[0])) @ (Y_train - u) + u
error_lt_plus_1 = fs
# print("Predictions:")
# print(fs)
# print()
# print("Ground Truth:")
# print(Y_test)
new_states_predicted = states_predicted[1:] + error_lt_plus_1
# error_old = np.absolute(states_predicted[1:] - states_t_plus_1)
# error_new = np.absolute(new_states_predicted - states_t_plus_1)
error_old = np.average(np.linalg.norm(states_predicted[1:] - states_t_plus_1, axis=1))
error_new = np.average(np.linalg.norm(new_states_predicted - states_t_plus_1, axis=1))
print()
if (error_new > error_old):
    print("GPR seems to have worked (new predictions closer to actual values than old predictions)")
else:
    print("GPR didn't work...")
# Leave the next 4 lines commented (5/1/2021)
# print("Average Error:")
# error = np.average(np.linalg.norm(fs - Y_test, axis=1))
# print(error)
# print((states_predicted + error) - linearized_states_t_plus_1)



    