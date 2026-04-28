import cv2
import numpy as np

# -------------------------
# 1. IMAGE PREPROCESSING
# -------------------------

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    img[:,:,0] *= (avg_gray / avg_b)
    img[:,:,1] *= (avg_gray / avg_g)
    img[:,:,2] *= (avg_gray / avg_r)
    
    return np.clip(img, 0, 255).astype(np.uint8)

# -------------------------
# 2. SCLERA SEGMENTATION
# -------------------------

def segment_sclera(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Sclera mask conditions
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    sclera = cv2.bitwise_and(img, img, mask=mask)
    
    return sclera, mask

# -------------------------
# 3. FEATURE EXTRACTION
# -------------------------

def extract_features(img, mask):
    pixels = img[mask > 0]
    
    if len(pixels) == 0:
        return None
    
    # RGB
    r_mean = np.mean(pixels[:,2])
    g_mean = np.mean(pixels[:,1])
    b_mean = np.mean(pixels[:,0])
    
    rg_ratio = r_mean / (g_mean + 1e-6)
    
    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask > 0]
    
    h_mean = np.mean(hsv_pixels[:,0])
    s_mean = np.mean(hsv_pixels[:,1])
    v_mean = np.mean(hsv_pixels[:,2])
    
    # Yellow hue range (20-40 degrees)
    yellow_hue_mask = (hsv_pixels[:,0] >= 20) & (hsv_pixels[:,0] <= 40)
    yellow_ratio = np.mean(yellow_hue_mask)
    
    # LAB (CRITICAL FOR YELLOW)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_pixels = lab[mask > 0]
    
    b_star_mean = np.mean(lab_pixels[:,2])   # yellow-blue axis
    b_star_std = np.std(lab_pixels[:,2])
    
    # Yellow threshold ratio (b* > 110 = strong yellow sclera)
    yellow_pixels_ratio = np.mean(lab_pixels[:,2] > 110)
    
    return [
        r_mean, g_mean, b_mean, rg_ratio,      # RGB
        h_mean, s_mean, v_mean, yellow_ratio,  # HSV + yellow hue
        b_star_mean, b_star_std, yellow_pixels_ratio  # LAB focused
    ]
