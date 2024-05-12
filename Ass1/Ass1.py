import itertools
import numpy as np
import tensorflow as tf
from scipy.stats import linregress


def generate_inputs(n):
    return list(itertools.product([0, 1], repeat=n))


def conv_int_to_binary_list(out_f,num_bits):
  
    return [int(bit) for bit in bin(out_f)[2:].zfill(num_bits)]


def is_linearly_separable(data, labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(data.shape[1],))
    ])
    
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.5,
        decay_steps=5,
        decay_rate=0.7
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels, epochs=150, verbose=0) #verbose=2 for each epoch data showing
    
    _, accuracy = model.evaluate(data, labels) #not storing loss
 
    return accuracy == 1.0


def find_rel_with_n(num_variables,cnt):
    slope, intercept, r_value, p_value, std_err = linregress(num_variables, cnt)
    print(f"Relation is: y = {slope:.2f}x + {intercept:.2f}")

# Main program
def main():
    cnt=[]
    num_variables_list = [] 
    for num_variables in range(2, 4):#should be (2,6) but its taking too long for 5
        count=0
        num_variables_list.append(num_variables)
        print(f"Checking binary functions with {num_variables} variables:")
        inputs = generate_inputs(num_variables)
        for out_fun in range(0, pow(2,pow(2,num_variables))):
            outputs = conv_int_to_binary_list(out_fun,pow(2,num_variables))
            data = np.array(inputs)
            labels = np.array(outputs)
            if is_linearly_separable(data, labels):
              # print("Linearly separable!")
               count=count+1
              # print("Not linearly separable.")
        cnt.append(count)
    print(f"No of linearly separable function list from 2 to 6: {cnt}")
    find_rel_with_n(num_variables_list,cnt)  #finding relation on input n

if __name__ == "__main__":
    main()
