print("Starting orchids2.py - DEBUG CHECK")
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import extcolors
from collections import Counter
import rembg
import subprocess
from inference_sdk import InferenceHTTPClient
import base64
with open("debug_log.txt", "w") as f:
    f.write("Script started\n")
# ========== CONFIGURATION ==========
DEFAULT_INPUT_IMAGE = "input.jpg"  # Default input image in current directory
DEFAULT_OUTPUT_DIR = "output"      # Default output directory

ROBOFLOW_KEYS = {
    'detection': "mhx8F4ixau4VZIgYD6XA",
    'pattern': "YAkiNGV1kA5VqHzUvfDr",
    'segmentation': "kLBXZ4Hp58tudq5TlIQZ"
}

# ========== MASKING FUNCTIONS ==========
def create_mask(input_path, output_dir, mask_num=1):
    """Enhanced masking with background removal and mask creation"""
    try:
        print(f"\n[DEBUG] Starting create_mask for {input_path}")  # TEST PRINT
        # Create output filename
        output_path = os.path.join(output_dir, f'Mask_{mask_num}.png')
        
        # Option 1: Use rembg if available
        try:
            input_image = Image.open(input_path)
            output_image = rembg.remove(input_image)
            output_image.save(output_path)
            print("Background removed using rembg")
        except Exception as e:
            # Option 2: Fallback to OpenCV-based background removal
            img = cv2.imread(input_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(output_path, mask)
            print("Background removed using OpenCV thresholding")

        # Create binary mask
        img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image from {output_path}")

        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA image
            mask = np.where(img[:, :, 3] > 0, 255, 0).astype(np.uint8)
        else:  # Grayscale image
            mask = np.where(img > 0, 255, 0).astype(np.uint8)

        # Save filled mask visualization
        filled_mask = fill_mask(mask)
        filled_path = os.path.join(output_dir, f'FilledMask_{mask_num}.png')
        cv2.imwrite(filled_path, filled_mask)
                # =============================================
        # INSERT THE PRINT STATEMENT RIGHT HERE:
        print("\n=== Masking Results ===")
        print(f"1. Transparent background saved to: {output_path}")
        print(f"2. Color-filled mask saved to: {filled_path}")
        print(f"3. Binary mask dimensions: {mask.shape}")
        # =============================================
        return {
            'mask_path': output_path,
            'filled_mask_path': filled_path,
            'binary_mask': mask
        }
        
    except Exception as e:
        print(f"Masking failed: {str(e)}")
        raise

def fill_mask(mask, color=(255, 0, 0)):
    """Creates a color-filled version of the mask"""
    color_filled = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_filled[mask > 0] = color
    return color_filled

# ========== IMAGE PROCESSING FUNCTIONS ==========
def preprocess_image(image_path, output_dir):
    """Enhanced preprocessing with masking support"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"\n[DEBUG] Starting preprocess_image for {image_path}")  # TEST PRINT
        
        print("\n=== Preprocessing Stage ===")
        print("1. Removing background and creating masks...")
        mask_results = create_mask(image_path, output_dir)
        
        # Verify mask results
        if not os.path.exists(mask_results['mask_path']):
            raise FileNotFoundError(f"Mask file not created: {mask_results['mask_path']}")
        
        print("\n2. Applying mask to original image...")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load original image")
            
        mask = cv2.imread(mask_results['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Failed to load mask image")
        
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        no_bg_path = os.path.join(output_dir, "no_bg.png")
        cv2.imwrite(no_bg_path, masked_img)
        print(f"- Background removed image saved to: {no_bg_path}")
        
        print("\n3. Enhancing image...")
        img_pil = Image.open(no_bg_path)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(1.2)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.1)
        enhanced_path = os.path.join(output_dir, "enhanced.png")
        img_pil.save(enhanced_path)
        print(f"- Enhanced image saved to: {enhanced_path}")
        
        return {
            'original_path': image_path,
            'mask_path': mask_results['mask_path'],
            'filled_mask_path': mask_results['filled_mask_path'],
            'no_bg_path': no_bg_path,
            'enhanced_path': enhanced_path
        }
        
    except Exception as e:
        print(f"\n! Preprocessing Error: {str(e)}")
        raise
print(f"\nCOlor converting")
def rgb_to_hex(r, g, b):
    """Convert RGB to HEX color format"""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV color space"""
    hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    return (h, round(s / 255 * 100, 2), round(v / 255 * 100, 2))  # Normalize S and V

def rgb_to_lab(r, g, b):
    """Convert RGB to LAB color space"""
    lab = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0]
    L, a, b = lab
    L = round((L / 255) * 100, 2)  # Normalize L to [0, 100]
    a = round(a - 128, 2)  # Normalize a to [-128, 128]
    b = round(b - 128, 2)  # Normalize b to [-128, 128]
    return (L, a, b)
def extract_colors(image_path, output_dir, numDok=1, tolerance=32, limit=10):
    """Enhanced color extraction with multiple color spaces"""
    try:
        print(f"\nColor Extracting")
        # Load and prepare image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file")
            
        image = cv2.resize(image, (512, 512))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Method 1: Using extcolors for dominant colors
        # colors_x = extcolors.extract_from_path(image_path, tolerance=tolerance, limit=limit)
        # colors, pixel_count = colors_x[0], colors_x[1]

        # # Filter out near-black colors
        # colors = [c for c in colors if not (c[0][0] < 20 and c[0][1] < 20 and c[0][2] < 20)]
        
        # Method 2: Pixel-level analysis excluding black
        pixels = image_rgb.reshape(-1, 3)
        mask = np.all(pixels != [0, 0, 0], axis=1)
        non_black_pixels = pixels[mask]
        
        # Calculate mean values
        mean_r, mean_g, mean_b = np.mean(non_black_pixels, axis=0).astype(int)
        
        # Prepare data for both methods
        
        pixel_data = {
            "R": non_black_pixels[:, 0],
            "G": non_black_pixels[:, 1],
            "B": non_black_pixels[:, 2],
            "HEX": [rgb_to_hex(r, g, b) for r, g, b in non_black_pixels],
            "HSV": [rgb_to_hsv(r, g, b) for r, g, b in non_black_pixels],
            "LAB": [rgb_to_lab(r, g, b) for r, g, b in non_black_pixels],
        }

        df_pixels = pd.DataFrame(pixel_data)

        # color_data = []
        # for color in pixels:
        #     rgb, count = color
        #     color_data.append({
        #         # 'type': 'dominant',
        #         'red': rgb[0],
        #         'green': rgb[1],
        #         'blue': rgb[2],
        #         'hex': rgb_to_hex(*rgb),
        #         'hsv': rgb_to_hsv(*rgb),
        #         'lab': rgb_to_lab(*rgb),
        #         # 'percentage': (count / pixel_count) * 100,
        #         # 'pixel_count': count
        #     })
        
        # Add mean color entry
        # color_data.append({
        #     'type': 'mean',
        #     'red': mean_r,
        #     'green': mean_g,
        #     'blue': mean_b,
        #     'hex': rgb_to_hex(mean_r, mean_g, mean_b),
        #     'hsv': rgb_to_hsv(mean_r, mean_g, mean_b),
        #     'lab': rgb_to_lab(mean_r, mean_g, mean_b),
        #     'percentage': 100 * len(non_black_pixels) / len(pixels),
        #     'pixel_count': len(non_black_pixels)
        # })
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"colors_{numDok}.csv")
        df_pixels.to_csv(csv_path, index=False)
        
        # Create color palette visualization
        # create_color_palette(color_data, output_dir)
        
        # Print summary
        print("\n=== Color Analysis Results ===")
        # print(f"- Found {len(colors)} dominant colors")
        print(f"- Mean color (excluding black): RGB({mean_r}, {mean_g}, {mean_b})")
        print(f"- Data saved to: {csv_path}")
        
        return csv_path
        
    except Exception as e:
        print(f"Color extraction error: {str(e)}")
        raise

# def create_color_palette(color_data, output_dir):
#     """Generate a visual color palette"""
#     try:
#         palette_height = 100
#         palette_width = len(color_data) * 100
#         palette = Image.new('RGB', (palette_width, palette_height))
        
#         x_offset = 0
#         for color in color_data:
#             color_block = Image.new('RGB', (100, 100), 
#                                  (color['red'], color['green'], color['blue']))
#             palette.paste(color_block, (x_offset, 0))
#             x_offset += 100
        
#         palette_path = os.path.join(output_dir, "color_palette.png")
#         palette.save(palette_path)
        
#     except Exception as e:
#         print(f"Palette creation error: {str(e)}")

def proportion_colro(image_path,output_path,numDok=1):
        # Extract colors and their proportions using the extract_color function
    colors_with_proportion = collect_color(image_path)

    # Print each color with its proportion
    for color, proportion in colors_with_proportion:
      if proportion > 0.01:
        print(f"Color: {color}, Proportion: {proportion:.2%}")

    # Save the extracted color data to a CSV file
    csv_filename = os.path.join(output_path, f'PropDok_{numDok}.csv')  # Create CSV filename based on the image filename

    color_data = [{'Red': color[0], 'Green': color[1], 'Blue': color[2], 'Proportion': proportion}
                  for color, proportion in colors_with_proportion]

    df = pd.DataFrame(color_data)
    df.to_csv(csv_filename, index=False)
    print(f'CSV file saved at: {csv_filename}')
    return csv_filename

def collect_color(image_path):
    colors, pixel_count = extcolors.extract_from_path(image_path)  # Extract colors using extcolors
    # Filter out black color (RGB: (0, 0, 0))
    colors = [color for color in colors if color[0] != (0, 0, 0)]

    total_pixels = sum(count for _, count in colors)  # Calculate total count of non-black pixels
    # Recalculate proportions
    colors_with_proportion = [(color, count / total_pixels) for color, count in colors]

    return colors_with_proportion  # Return list of colors and their proportions

def classify_pattern(image_path, output_dir):
    """Classify orchid pattern"""
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_KEYS['pattern']
    )
    
    result = client.run_workflow(
        workspace_name="cher-final-senior-project",
        workflow_id="classify-and-conditionally-detect-5",
        images={"image": image_path},
        use_cache=True
    )
    
    # Save results
    csv_path = os.path.join(output_dir, "pattern.csv")
    if result and isinstance(result, list):
        predictions = result[0].get('classification_predictions', {}).get('predictions', [])
        pd.DataFrame(predictions).to_csv(csv_path, index=False)
    return csv_path



def radial_analysis(image_path, output_dir):
    """Complete radial analysis with debug outputs"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Save intermediate images
        cv2.imwrite(os.path.join(output_dir, "radial_binary.png"), binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        target_class = "column"
        detected_objects = 0
        cx, cy = 0, 0
        angle_mem = 0

        data = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            if 5 <= len(polygon) <= 12 and cv2.contourArea(polygon) > 100:
                detected_objects += 1
                cv2.drawContours(img, [polygon], -1, (0, 255, 0), 2)

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                    
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Create visualization
                vis = img.copy()
                cv2.drawContours(vis, [contour], -1, (0,255,0), 2)
                cv2.circle(vis, (cx,cy), 5, (0,0,255), -1)
                
                M = cv2.moments(polygon)
                if M["m00"] != 0:

                                            # Compute the centroid for the specific class
                    if target_class in df["Class"].unique():
                            centroid = df[df["Class"] == target_class][["X", "Y"]].mean()
                            print(f"Centroid of class '{target_class}': ({centroid.X}, {centroid.Y})")
                            cx = centroid.X
                            cy = centroid.Y

                    else:
                            print(f"Class '{target_class}' not found in the dataset.")

                    if cx == 0 and cy == 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])


                    # cv2.circle(result_image, (cx, cy), 5, (255, 0, 255), -1)
                    points = [(int(pt[0][0]), int(pt[0][1])) for pt in polygon]

                    for j in range(len(points)):
                        p1 = points[j]
                        p2 = points[(j + 1) % len(points)]

                        v1 = (cx - p1[0], cy - p1[1])
                        v2 = (cx - p2[0], cy - p2[1])
                        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

                        if mag1 == 0 or mag2 == 0:
                            continue  # Avoid division by zero

                        dot = v1[0] * v2[0] + v1[1] * v2[1]
                        cos_theta = dot / (mag1 * mag2)
                        angle = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))

                        if angle < 30:
                            angle_mem = angle + angle_mem
                            continue
                        else:
                            angle = angle_mem + angle
                            angle_mem = 0
                        # Store data in a list
                        data.append({"P1": p1, "P2": p2, "Centroid": (cx, cy), "Angle": round(angle, 2)})

                        # Draw on image
                        # cv2.line(result_image, (cx, cy), p1, (45, 45, 0), 1)
                        # cv2.line(result_image, (cx, cy), p2, (60, 60, 0), 1)
                        cv2.putText(img, f"{angle:.1f}°", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


                # Save contour visualization
                cv2.imwrite(os.path.join(output_dir, f"contour_{len(data)}.png"), vis)
        
        # Save results
        csv_path = os.path.join(output_dir, "radial.csv")
        if data:
            pd.DataFrame(data).to_csv(csv_path, index=False)
        else:
            # Create empty file with error message if no contours
            with open(csv_path, 'w') as f:
                f.write("error,no_valid_contours_detected\n")
                f.write(f"contours_found,{len(contours)}\n")
                f.write(f"min_area_threshold,100\n")
        
        return csv_path
        
    except Exception as e:
        error_path = os.path.join(output_dir, "radial_error.txt")
        with open(error_path, 'w') as f:
            f.write(f"Radial analysis failed: {str(e)}\n")
        raise
    
def detect_orchids(image_path, output_dir):
    """Detect orchids in image using Roboflow"""
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_KEYS['detection']
    )
    
    result = client.run_workflow(
        workspace_name="cher-ver3",
        workflow_id="detect-count-and-visualize",
        images={"image": image_path},
        use_cache=True
    )
    
    # Process results
    data = []
    if "predictions" in result[0]["predictions"]:
        for pred in result[0]["predictions"]["predictions"]:
            data.append({
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
                "confidence": pred["confidence"],
                "class": pred["class"]
            })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "detection.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path

def perform_segmentation(image_path, output_dir):
    """Perform image segmentation"""
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=ROBOFLOW_KEYS['segmentation']
    )
    
    result = client.run_workflow(
        workspace_name="cher-ver2-u98yx",
        workflow_id="detect-count-and-visualize",
        images={"image": image_path},
        use_cache=True
    )
    
    # Save results
    csv_path = os.path.join(output_dir, "segmentation.csv")
    if "predictions" in result[0]:
        data = [{
            "x": p["x"],
            "y": p["y"],
            "class": p["class"]
        } for p in result[0]["predictions"].get("predictions", [])]
        pd.DataFrame(data).to_csv(csv_path, index=False)
    
    return {'csv_path': csv_path}


    

def process_image(image_path, output_dir):
    """Complete image processing pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"\n=== Starting Analysis of {os.path.basename(image_path)} ===")
        
        # 1. Preprocessing
        processed = preprocess_image(image_path, output_dir)
        
        print("\n=== Detection ===")
        detection_csv = detect_orchids(processed['enhanced_path'], output_dir)
        print(f"- Radial measurements saved to: {detection_csv}")
        print(f"- Contour visualizations: output/contour_*.png")
        
        # 2. Color Extraction
        print("\n=== Extraction ===")
        color_csv = extract_colors(processed['enhanced_path'], output_dir)
        print(f"- Main colors saved to: {color_csv}")
        print(f"- Color palette visualization: output/color_palette.png")
        
        #3.Colro Analysis
        print("\n=== Proportion ===")
        color_csv = proportion_colro(image_path,output_dir,numDok=1)
        print(f"- Main colors saved to: {color_csv}")
        print(f"- Color palette visualization: output/color_palette.png")
        
        # 4. Radial analysis
        print("\n=== Shape Analysis ===")
        radial_csv = radial_analysis(processed['enhanced_path'], output_dir)
        print(f"- Radial measurements saved to: {radial_csv}")
        print(f"- Contour visualizations: output/contour_*.png")
        
        print("\n=== Classify Pattern ===")
        pattern_csv = classify_pattern(processed['enhanced_path'], output_dir)
        print(f"- Radial measurements saved to: {pattern_csv}")
        print(f"- Contour visualizations: output/contour_*.png")
        
        print("\n=== Classify Pattern ===")
        segmentataion_csv = perform_segmentation(processed['enhanced_path'], output_dir)
        print(f"- Radial measurements saved to: {segmentataion_csv}")
        print(f"- Contour visualizations: output/contour_*.png")

        
        print("\n=== Processing Complete ===")
        print(f"All results saved to: {os.path.abspath(output_dir)}")
        print("\nMasking Results Available:")
        print(f"- Transparent BG: {processed['mask_path']}")
        print(f"- Color Mask: {processed['filled_mask_path']}")
        print(f"- No BG Image: {processed['no_bg_path']}")
        
    except Exception as e:
        print(f"\n! Processing Failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Orchid Feature Analyzer')
    parser.add_argument('input', nargs='?', default=DEFAULT_INPUT_IMAGE, help='Input image path')
    parser.add_argument('output', nargs='?', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"\nError: Input file '{args.input}' not found")
        print("Please either:")
        print(f"1. Place an image named '{DEFAULT_INPUT_IMAGE}' in current directory")
        print("2. Specify custom paths: python orchids.py <input_path> <output_dir>")
    else:
        process_image(args.input, args.output)
