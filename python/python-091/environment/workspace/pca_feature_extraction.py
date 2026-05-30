
import numpy as np
from typing import Tuple, List


def compute_mean_face(images: np.ndarray) -> np.ndarray:
    return np.mean(images, axis=0)


def compute_pca_vectors(images: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = images.shape
    
    if n_components is None:
        n_components = min(N - 1, d)
    
    n_components = min(n_components, N - 1, d)
    
    mean_face = compute_mean_face(images)
    

    A = images - mean_face
    

    L = A @ A.T
    

    eigenvalues_full, eigenvectors_full = np.linalg.eigh(L)
    

    idx = np.argsort(eigenvalues_full)[::-1]
    eigenvalues_full = eigenvalues_full[idx]
    eigenvectors_full = eigenvectors_full[:, idx]
    

    eigenvalues = eigenvalues_full[:n_components]
    


    eigenvectors = np.zeros((n_components, d))
    for i in range(n_components):
        if eigenvalues[i] > 1e-14:
            v = A.T @ eigenvectors_full[:, i]
            v = v / np.linalg.norm(v)
            eigenvectors[i] = v
    
    return eigenvectors, eigenvalues, mean_face


def project_image(image: np.ndarray, eigenvectors: np.ndarray,
                  mean_face: np.ndarray) -> np.ndarray:
    centered = image - mean_face
    coefficients = eigenvectors @ centered
    return coefficients


def reconstruct_image(coefficients: np.ndarray, eigenvectors: np.ndarray,
                      mean_face: np.ndarray) -> np.ndarray:
    return mean_face + coefficients @ eigenvectors


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    diff = original - reconstructed
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    

    max_val = np.max(np.abs(original))
    if max_val > 1e-14:
        psnr = 20.0 * np.log10(max_val / rmse)
    else:
        psnr = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'psnr': float(psnr),
        'max_error': float(np.max(np.abs(diff))),
        'mean_error': float(np.mean(np.abs(diff)))
    }


def generate_synthetic_bscans(n_images: int = 50, n_samples: int = 256,
                              n_lines: int = 64) -> np.ndarray:
    image_size = n_samples * n_lines
    images = np.zeros((n_images, image_size))
    
    for img_idx in range(n_images):
        bscan = np.zeros((n_lines, n_samples))
        

        n_interfaces = np.random.randint(2, 6)
        for _ in range(n_interfaces):
            depth = np.random.randint(20, n_samples - 20)
            amplitude = np.random.uniform(0.3, 1.0)
            thickness = np.random.randint(2, 8)
            
            for line in range(n_lines):

                depth_variation = int(3 * np.sin(2 * np.pi * line / n_lines + img_idx))
                actual_depth = depth + depth_variation
                actual_depth = max(0, min(n_samples - 1, actual_depth))
                
                for t in range(thickness):
                    if actual_depth + t < n_samples:
                        bscan[line, actual_depth + t] += amplitude * np.exp(-t**2 / 4.0)
        

        noise_level = np.random.uniform(0.05, 0.15)
        bscan += noise_level * np.random.randn(n_lines, n_samples)
        
        images[img_idx] = bscan.flatten()
    
    return images


def pca_bscan_analysis(n_images: int = 50, n_components: int = 10) -> dict:
    images = generate_synthetic_bscans(n_images)
    
    eigenvectors, eigenvalues, mean_face = compute_pca_vectors(images, n_components)
    

    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)
    variance_ratio = eigenvalues / (total_variance + 1e-14)
    cumulative_ratio = cumulative_variance / (total_variance + 1e-14)
    

    test_image = images[0]
    coeffs = project_image(test_image, eigenvectors, mean_face)
    reconstructed = reconstruct_image(coeffs, eigenvectors, mean_face)
    error_info = compute_reconstruction_error(test_image, reconstructed)
    
    return {
        'n_images': n_images,
        'n_components': n_components,
        'eigenvalues': eigenvalues.tolist(),
        'variance_ratio': variance_ratio.tolist(),
        'cumulative_variance_ratio': cumulative_ratio.tolist(),
        'reconstruction_error': error_info
    }
