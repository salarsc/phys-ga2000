import numpy as np
#we record the number here
num= 100.98763
print("the actaul number is ", num)

#we define the function to give array of bits for the input
def get_bits(number):
  
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return np.array(list(reversed(bits)), dtype=int)


#we turn into a floating 32 precition numpy number
num32=np.float32(num)

#we turn i into bites
bit32=get_bits(num32)

#print the output
print("the numpy floating point numer in binary(32 float is) ",bit32)

#we use the standard equation of transformation of bits to float and based on it we definethe following analytical calculation function 
def floatmaker (binary):
    
    
    sign_bit = int(binary[0])
    exp_bits = binary[1:9]
    manti_bits = np.insert(binary[9:], 0, int(1))
    
#we now turn th exp part to a real exp number
    exponent = int(''.join(map(str, exp_bits)), 2) - 127
      
# we calculate the mantisa
    mantissa = 0
    for i, bit in enumerate(manti_bits):
        mantissa += int(bit) * (2 ** -i)
    
    # Calculate the final float number
    floatn = ((-1) ** sign_bit) * mantissa * (2 ** exponent)
    
    return floatn

# Example usage:
binary_list = bit32
result = floatmaker(binary_list)
print("analytical precise float number of the above binary is:", result)

difference=result-num

print("tthe difference is:", difference)

