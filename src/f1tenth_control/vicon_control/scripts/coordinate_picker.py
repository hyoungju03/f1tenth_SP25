# save this code as place_points.py
import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# List to store points
points = []

# Define the event for mouse click
def onclick(event):
    # Check if we have already selected 4 points
    if len(points) < 4:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # Plot the point on the image
        plt.scatter(x, y, color="red")
        plt.draw()

        # If 4 points are selected, disconnect the event listener
        if len(points) == 4:
            print("4 points selected. Closing the plot.")
            plt.disconnect(cid)
            plt.close()

# Show the image
fig, ax = plt.subplots()
ax.imshow(image)

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Display the image for point selection
print("Click on 4 points on the image.")
plt.show()

# Output the final coordinates
print("Selected points:", points)

