from sympy import integrate, exp, symbols, oo, latex

# Define symbols
x, y, z, lam = symbols('x y z lambda', positive=True)

# Joint PDF
f = lam**2 * exp(-lam * (x + y))

# Calculate P(X - Y > z) for z >= 0
p_x_y_greater_z = integrate(f, (x, y + z, oo), (y, 0, oo))

# CDF = 1 - P(X - Y > z)
cdf_z_pos = 1 - p_x_y_greater_z

# Print the result in LaTeX format
print("CDF for z >= 0: P(X - Y <= z) =")
print(latex(cdf_z_pos))