import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import random

# Load the YOLO model
model = YOLO('YOLO_Model.pt')

# Streamlit app title
st.title("YOLO Vehicle Detection with KMeans Clustering")

# User input for the number of clusters
n_clusters = st.number_input("Enter the number of clusters:", min_value=2, max_value=10, value=4)

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Apply YOLO model for object detection
    results = model.predict(image)

    centroids = []
    bbox_info = []

    for result in results:
        for obj in result.boxes:
            bbox = obj.xyxy[0].cpu().numpy()  # Bounding box coordinates
            class_id = int(obj.cls[0].cpu().numpy()) if obj.cls is not None else -1  # Class ID
            conf = obj.conf[0].cpu().numpy() if obj.conf is not None else 0.0  # Confidence score
            
            # Convert bbox coordinates to integers
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroids.append([cx, cy])
            bbox_info.append((x1, y1, x2, y2))  # Store bbox info

    if centroids:
        # Apply KMeans clustering
        centroids = np.array(centroids)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(centroids)
        labels = kmeans.labels_

        # Generate unique colors for each cluster
        cluster_colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(n_clusters)}

        # Count the number of cars in each cluster
        cluster_counts = {i: 0 for i in range(n_clusters)}
        for label in labels:
            cluster_counts[label] += 1

        # Draw bounding boxes and cluster points with cluster-specific colors
        for idx, (x1, y1, x2, y2) in enumerate(bbox_info):
            cluster_label = labels[idx]
            color = cluster_colors[cluster_label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            cx, cy = centroids[idx]
            cv2.circle(image, (cx, cy), 5, color, -1)  # Mark cluster points

        # Display the cluster counts on the image
        y_offset = 50
        for label, count in cluster_counts.items():
            text = f'Cluster {label}: {count} cars'
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 40

        # Display the processed image
        st.image(image, caption='Processed Image', use_column_width=True)
