from scipy.optimize import minimize


# Lambda function to minimize
f = lambda x:(x[0]-0.5)**2+(x[1]-0.5)**2

# Initial guess
x0=[0.1,0.1]

# Perform minimization
result = minimize(f, x0)

print(result.x)      # Minimum point (approx)
print(result.fun)    # Minimum value of f




# Lambda function to minimize
f = lambda x,y:(x-0.5)**2+(y-0.5)**2

# Initial guess
x0=[0.1,0.1]

# Perform minimization
result = minimize(f, x0)

print(result.x)      # Minimum point (approx)
print(result.fun)    # Minimum value of f
