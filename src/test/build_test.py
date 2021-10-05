import unittest
import numpy as np
import os
import io
from PIL import Image, ImageFile
from src.rembg.u2net.detect import load_model
from src.rembg.bg import remove
from torch.nn import Module
ImageFile.LOAD_TRUNCATED_IMAGES = True


def calculate_histo_similarity(layer_1:np.ndarray, layer_2:np.ndarray):
    histo_count_1, histo_dist_1 = np.histogram(layer_1.flatten(), bins=128)
    histo_count_2, histo_dist_2 = np.histogram(layer_2.flatten(), bins=128)
    count_similarity = (histo_count_1.T @ histo_count_2) / \
                       (np.linalg.norm(histo_count_1) * np.linalg.norm(histo_count_2))
    dist_similarity = (histo_dist_1.T @ histo_dist_2) / \
                      (np.linalg.norm(histo_dist_1) * np.linalg.norm(histo_dist_2))
    return count_similarity, dist_similarity


def generate_result(source_file, target_file, alpha_matting=False):
    test = remove(source_file, alpha_matting=alpha_matting)
    test = np.array(Image.open(io.BytesIO(test)).convert("RGBA"))
    target = np.array(Image.open(io.BytesIO(target_file)).convert("RGBA"))
    return test, target


class TestEnvironmentalCase(unittest.TestCase):
    def test_image_support(self):
        pass

    def test_mlhub_support(self):
        try:
            import mlhub.pkg
            import mlhub.utils
        except:
            self.fail('MLHub package is current unavailable')
        self.assertIn('mlask', dir(mlhub.pkg))
        self.assertIn('mlcat', dir(mlhub.pkg))
        self.assertIn('get_cmd_cwd', dir(mlhub.pkg))
        self.assertIn('yes_or_no', dir(mlhub.utils))
        self.assertIn('get_package_dir', dir(mlhub.utils))

    def test_load_model(self):
        net = load_model()
        self.assertIsInstance(net, Module)

    def test_load_data(self):
        f_1 = np.array(Image.open('examples/animal-1.jpg'))
        f_2 = np.array(Image.open('examples/animal-1.out.png'))
        self.assertEqual(f_1.shape, (667,1000,3))
        self.assertEqual(f_2.shape, (667,1000,4))


class TestFunctionalCases(unittest.TestCase):
    def test_remove_whole(self):
        test_result, target_result = generate_result(np.fromfile(os.path.join('examples', "animal-1.jpg")),
                                                     np.fromfile(os.path.join('examples', "animal-1.out.png")))
        msg = 'Assertion failed because the output data does not match the target'
        count_similarity, dist_similarity = calculate_histo_similarity(test_result, target_result)
        self.assertGreaterEqual(count_similarity, 0.99, msg)
        self.assertGreaterEqual(dist_similarity, 0.99, msg)

    def test_remove_by_layer(self):
        test_result, target_result = generate_result(np.fromfile(os.path.join('examples', "car-3.jpg")),
                                                     np.fromfile(os.path.join('examples', "car-3.out.png")))
        msg = 'Assertion failed because the output data of layer {} does not match the target'
        for i in range(test_result.shape[2]):
            count_similarity, dist_similarity = calculate_histo_similarity(test_result[i], target_result[i])
            self.assertGreaterEqual(count_similarity, 0.99, msg.format(i+1))
            self.assertGreaterEqual(dist_similarity, 0.99, msg.format(i+1))

    def test_alpha_matting(self):
        try:
            from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
            from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
            from pymatting.util.util import stack_images
            from scipy.ndimage.morphology import binary_erosion
        except:
            self.fail('Alpha matting is currently unavailable on this system')
        test_result, target_result = generate_result(np.fromfile(os.path.join('examples', "food-1.jpg")),
                                                     np.fromfile(os.path.join('examples', "food-1.out.jpg")),
                                                     alpha_matting=True)
        msg = 'Assertion failed because the output data of layer {} does not match the target'
        for i in range(test_result.shape[2]):
            count_similarity, dist_similarity = calculate_histo_similarity(test_result[i], target_result[i])
            self.assertGreaterEqual(count_similarity, 0.99, msg.format(i+1))
            self.assertGreaterEqual(dist_similarity, 0.99, msg.format(i+1))

if __name__ == '__main__':
    unittest.main()
