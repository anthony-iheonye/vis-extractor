import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.color import rgb2lab, rgb2gray
from skimage.feature.texture import (graycomatrix, graycoprops)
from skimage.measure import (label, regionprops_table, shannon_entropy)
from skimage.segmentation import clear_border

from app.services import ImageAndMaskDatasetBuilder
from app.services.visual_attributes import VisualPropertiesCalibrator
from app.utils import create_directory


class VisualPropertiesExtractor:
    """Class for extracting visual attributes (size, shape, color, texture) of segmented objects within RGB images."""

    def __init__(self,
                 start_image_index: int,
                 images_directory: str,
                 masks_directory: str,
                 image_mask_channels: Tuple[int, int],

                 exclude_partial_objects: bool = False,
                 cache_visual_attributes: bool = False,
                 calibrator: VisualPropertiesCalibrator = None,

                 image_group_size: int = 1,
                 interval_per_image_group: int = 15,
                 initial_group_interval: int = 0,

                 save_location: str = None,
                 overwrite_existing_record: bool = True,
                 realtime_update: bool = False,
                 image_cache_directory: Optional[str] = None,
                 ):
        """
        Extracts visual attributes (size, shape, color, texture) of segmented objects (e.g., particles, fruits, cells) in RGB images using their corresponding masks.

        :param start_image_index: (int) The numeric label to assign to the first image in the dataset. This index determines how each image is referenced in the visual attribute report (e.g., "img_5", "img_6", ...) but does not affect dataset loading or ordering. Useful when resuming processing or aligning with external logs.
        :param images_directory: (str) Path to the directory where the images are stored.
        :param masks_directory: (str) Path to the directory where the masks are stored.
        :param image_mask_channels: (tuple) A tuple containing the number of image and mask channels (e.g. (3, 1) for rgb image and grayscale mask, or (3, 3) for a rgb image and mask).
        :param exclude_partial_objects: (bool) If True, objects that touch the image border are excluded from computation.
        :param cache_visual_attributes: (bool) If True, enables caching of visual properties during processing. It may improve performance but uses more memory.
        :param calibrator: (VisualPropertiesCalibrator) An instance used to calibrate both color and size measurements. If provided, the extractor converts camera
        LAB values to match those of a physical colorimeter, and scales size-related features (e.g., area, perimeter) from pixels to metric units using predefined
        factors (e.g., mm² for area).
        :param image_group_size: (int) Number of images captured per group (e.g., per drying interval). Visual attributes are averaged across each group.
        :param interval_per_image_group: (int) The time interval (in minutes or seconds) between the capture of each group of images.This value is used to increment the time label assigned to the average visual attributes of each group. For example, if set to 15 and the initial group interval is 0, then groups will be labeled as time = 0, 15, 30, etc.
        :param initial_group_interval: (int) Starting value for time labeling of the first group (e.g., time=0 for the first set). Useful for continuing processing from a specific stage.
        :param save_location: (str) Directory where computed visual attributes will be saved in JSON and CSV formats. If None, nothing is saved to disk.
        :param overwrite_existing_record: (bool)  If True, overwrites existing saved visual property files at the `save_location`.
        :param realtime_update: If True, the JSON/CSV files are updated in real time as images are processed.
        Enables streaming writes but increases processing time.
        :param image_cache_directory:  Directory for caching image-mask data used during preprocessing.
        """

        # Image and mask directories
        self.image_directory = images_directory
        self.mask_directory = masks_directory
        self.realtime_update = tf.constant(value=realtime_update, dtype=tf.bool)

        # Generate image-mask dataset
        dataset_builder = ImageAndMaskDatasetBuilder(images_directory=images_directory,
                                                     masks_directory=masks_directory,
                                                     image_mask_channels=image_mask_channels,
                                                     cache_directory=image_cache_directory)
        dataset_builder.run()
        self.image_mask_dataset = dataset_builder.image_mask_dataset

        self.image_index = start_image_index
        self.tune = tf.data.AUTOTUNE

        # The base visual properties for computing shrinkage and color indices. By default, it is set to
        # ('eccentricity', 'equivalent_diameter', 'feret_diameter_max', 'filled_area', 'perimeter',
        # 'mean_intensity', 'intensity_image') more properties can be found on:
        # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        self.size_color_props = ('label', 'eccentricity', 'equivalent_diameter', 'feret_diameter_max',
                                 'filled_area', 'perimeter', 'mean_intensity', 'intensity_image')

        # The textural properties to be computed. values can be any/some/all the following: 'contrast',
        # 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'.
        # To compute uniformity, 'ASM' must be included in the list.
        self.texture_properties = ('contrast', 'correlation', 'energy', 'homogeneity', 'ASM')

        # conversion factors for area (mm^2/pixel), equivalent diameter (mm/pixel) and perimeter (mm/pixel)
        self.size_factor = calibrator.size_factor

        # The number of images that make up a group (for computing average visual attribute)
        self.image_group_size = image_group_size

        # information that would be used to group the visual properties by interval
        if self.image_group_size > 1 and interval_per_image_group:
            self.group_images_by_time = tf.constant(True)
            self.capture_interval = interval_per_image_group
            self.current_interval = initial_group_interval
            self.grouped_props_df = None
        else:
            self.current_interval = None
            self.group_images_by_time = tf.constant(False)

        self.exclude_partial_objects = exclude_partial_objects
        self.cache_visual_attributes = cache_visual_attributes

        if save_location:
            self.save_location = create_directory(save_location, return_dir=True, overwrite_if_existing=False)
            self.comprehensive_props_filepath = os.path.join(self.save_location, 'comprehensive_visual_properties.json')
            self.mean_props_filepath = os.path.join(self.save_location, 'mean_visual_properties.json')
            if self.group_images_by_time:
                self.grouped_props_filepath = os.path.join(self.save_location, 'visual_props_per_time_interval.json')
        else:
            self.save_location = None

        self.overwrite_existing_record = overwrite_existing_record

        self.comprehensive_props_df = pd.DataFrame()
        self.mean_props_df = pd.DataFrame()
        self.temp_comprehensive_props_df = None
        self.temp_mean_props_df = None
        self.temp_intensity_image_dict = dict()


        self.calibrator = calibrator

        if isinstance(calibrator, VisualPropertiesCalibrator):
            self.calibrate_color = True
            self.compute_actual_size = True
        else:
            self.calibrate_color = False
            self.compute_actual_size = False

        self._check_for_saved_visual_properties_json_file()

    def _produce_regionprops_table(self, rgb_image, mask):
        """
        Produces a region properties table

        :param rgb_image: rgb image
        :param mask: mask
        :return: Dictionary mapping property names to an array of values of that property, one value per region. This
            dictionary can be used as input to pandas `DataFrame` to map property names to columns in the frame and
            regions to rows. If the image has no regions, the arrays will have length 0, but the correct type.
        """
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        mask = np.where(mask.numpy() > 0, True, False)
        # produce label of the mask
        mask_label = label(label_image=mask, connectivity=1)
        if self.exclude_partial_objects:
            mask_label = clear_border(labels=mask_label)

        # Extract region properties for each segmented object
        temp_properties_dict = regionprops_table(label_image=mask_label,
                                                 intensity_image=rgb_image.numpy(),
                                                 properties=self.size_color_props,
                                                 cache=self.cache_visual_attributes)

        # Transfer the intensity images from the temp_properties_dict to a new dictionary temp_intensity_image_dict
        self.temp_intensity_image_dict['intensity_image'] = temp_properties_dict.pop('intensity_image')

        # create a new dataframe containing properties within the temp_properties_dict
        temp_properties_dataframe = pd.DataFrame(temp_properties_dict)

        return temp_properties_dataframe

    @staticmethod
    def _compute_roundness(area: float, perimeter: float):
        """"
        Computes the roundness of a segmented object based on its area and perimeter.

        :param area: area of the object
        :param perimeter: perimeter of the object
        """
        quotient = area / (perimeter ** 2)
        roundness = 4 * np.pi * quotient
        return roundness

    @staticmethod
    def _compute_shannon_entropy(image, base: int = 2):
        """
        computes the Shannon entropy (S) of an image using the formula:
        S = -sum(pk * log(pk)), where pk are frequency/probability of pixels of value k.

        :param image: (2D numpy array) grayscale Image
        :param base: (float) The logarithmic base to use. The value in the
            literature was base 2.
        :return: (float) entropy
        """
        entropy = shannon_entropy(image=image, base=base)
        return entropy

    def _compute_texture(self, gray_image, distances: tuple = (1,), angles: tuple = (0,), levels: int = 256):
        """
        Computes the textural parameters of a grayscale image

        :param gray_image: (2-D ndarray) uint8 grayscale image [0 - 225]
        :param distances: (list) eg. [0, 1] 0 is the number of rows between the pixel of interest and its neighbours, and 1
            is the number of columns (to the right) between the pixel of interest and its neighbours. with the first value
            '0', the neighboring pixel to the right of the pixel of interest, will be on the same row. on the other hand,
            with the second value being '1', it means that the neighboring pixel would be on the next column to the right.
            If distances is set to [0], then the neighbour is the pixel location to the right and on the same row as the
            the pixel of interest.
        :param angles: (tuple) eg if angles were [0, np.pi/2], to located of the neighboring we move be zero degrees to the
            right of the pixel of interest, then 90 degrees upwards.
        :param levels: (tuple) The number of level 'bards' into which the intensities of the grayscale image. The value should be
            the highest pixel intensity in the image. for uint8 grayscale image, the lowest levels value is 255.
            grey-levels counted (typically 256 for an 8-bit image). The maximum value is 256.
        :return: (4-D ndarray) The grey-level co-occurrence histogram. The value P[i,j,d,theta] is the number of times that
            grey-level j occurs at a distance d and at an angle theta from grey-level i. If normed is False, the output is
            of type uint32, otherwise it is float64.
        """

        # produce grey level co-occurence matrix (a histogram of co-occurring greyscale values at a given offset
        # over an image)
        glc_matrix = graycomatrix(image=gray_image, distances=list(distances), angles=list(angles),
                                  levels=levels)
        grey_properties = {prop: graycoprops(P=glc_matrix, prop=prop)[0][0] for prop in
                           sorted(list(self.texture_properties))}

        grey_properties['entropy'] = self._compute_shannon_entropy(gray_image)

        if 'ASM' in self.texture_properties:
            grey_properties['uniformity'] = grey_properties.pop('ASM')
        return grey_properties

    def _produce_texture_dataframe(self, intensity_image_dict: dict):
        """
        Produces a pandas dataframe containing textural values for all the object of interest in the rgb image.

        :param intensity_image_dict: (dict) A dict containing image of each object of interest inside the region bounding box.
            In order to compute the textural values, each would be converted to grayscale before computation is carried
            out.
        """
        texture = pd.DataFrame()

        for idx, image in enumerate(intensity_image_dict['intensity_image']):
            gray_image = (rgb2gray(image) * 255).astype(np.uint8)
            texture_values = self._compute_texture(gray_image=gray_image, distances=(1,))

            texture_values = pd.DataFrame(data=texture_values, index=[0, ])
            # texture = texture.append(texture_values, ignore_index=True)
            # print(texture)
            texture = pd.concat([texture, texture_values], axis='index', ignore_index=True)

        return texture

    @staticmethod
    def _convert_rgb2lab(red_intensity: float, green_intensity: float, blue_intensity: float):
        """
        convert a color in rgb color space to Lab color space

        :param red_intensity: rgb color of the red channel
        :param green_intensity: rgb color of the green channel
        :param blue_intensity: rgb color of the blue channel
        :return:
        """
        rgb_color = np.asarray([red_intensity, green_intensity, blue_intensity], dtype=np.float32) / 255
        l_index, a_index, b_index = rgb2lab(rgb_color)
        return l_index, a_index, b_index

    def _produce_lab_panda_series(self, red_intensities: pd.Series, green_intensities: pd.Series,
                                  blue_intensities: pd.Series):
        """
        Produce three pandas series object, one for the L* (lightness index), a* (greenish/redness index) and b*
        (blueish/yellowish index)

        :param red_intensities: (panda series) Series containing the rgb red intensities
        :param green_intensities: (panda series) Series containing the rgb green intensities
        :param blue_intensities: (panda series) Series containing the rgb blue intensities
        :return: (pandas Series) Series for L*, a* and b* color values
        """
        l = []
        a = []
        b = []
        for idx, (r_col, g_col, b_col) in enumerate(zip(red_intensities, green_intensities, blue_intensities)):
            lab = self._convert_rgb2lab(r_col, g_col, b_col)
            l.append(lab[0])
            a.append(lab[1])
            b.append(lab[2])
        return l, a, b

    def _check_for_saved_visual_properties_json_file(self):
        """
        checks if there exists json files that contain the comprehensive and mean properties from previous saved
        images.
        """
        if self.save_location is not None:
            if os.path.exists(self.comprehensive_props_filepath) and os.path.exists(self.mean_props_filepath) \
                    and self.overwrite_existing_record is False:
                # copy the contents to dataframes
                self.comprehensive_props_df = pd.read_json(self.comprehensive_props_filepath)
                self.mean_props_df = pd.read_json(self.mean_props_filepath)

            elif os.path.exists(self.comprehensive_props_filepath) and os.path.exists(self.mean_props_filepath) \
                    and self.overwrite_existing_record is True:
                print(f"Previously saved data on visual properties (JSON file) at {self.save_location} would be "
                      f"overwritten.")
                self.comprehensive_props_df = pd.DataFrame()
                self.mean_props_df = pd.DataFrame()

        else:
            print(f"No previously saved data (JSON file) on visual properties exist.")

    def _update_parent_dataframes(self):
        """
        Updates the saved comprehensive and mean visual properties json files, as well as save a csv copy of the
        file.
        """
        # update the mean and comprehensive visual properties dataframes
        # self.comprehensive_props_df = self.comprehensive_props_df.append(other=self.temp_comprehensive_props_df,
        #                                                                  ignore_index=True)
        # self.mean_props_df = self.mean_props_df.append(other=self.temp_mean_props_df, ignore_index=True)

        self.comprehensive_props_df = pd.concat([self.comprehensive_props_df, self.temp_comprehensive_props_df],
                                                axis='index', ignore_index=True)
        self.mean_props_df = pd.concat([self.mean_props_df, self.temp_mean_props_df], axis='index', ignore_index=True)

    def _save_dataframe_to_json_and_csv(self):
        # update json files
        self.comprehensive_props_df.to_json(path_or_buf=self.comprehensive_props_filepath)
        self.mean_props_df.to_json(path_or_buf=self.mean_props_filepath)

        # save a csv copy
        self.comprehensive_props_df.to_csv(path_or_buf=self.comprehensive_props_filepath.replace('json', 'csv'))
        self.mean_props_df.to_csv(path_or_buf=self.mean_props_filepath.replace('json', 'csv'))

        # Compute the average visual attributes for each group of images captured at a specific time interval
        if self.group_images_by_time:
            time_grp = self.mean_props_df.groupby(by=['time'])
            numeric_columns = self.mean_props_df.select_dtypes(include='number').columns

            self.grouped_props_df = time_grp[numeric_columns].mean()
            self.grouped_props_df.to_json(path_or_buf=self.grouped_props_filepath)
            self.grouped_props_df.to_csv(path_or_buf=self.grouped_props_filepath.replace('json', 'csv'))

    def _compute_visual_properties_for_single_image_and_mask(self, counter, image_index, rgb_image, mask):
        """Computes the visual properties of object of interest in a rgb image."""
        temp_props_df = self._produce_regionprops_table(rgb_image=rgb_image, mask=mask)

        if self.compute_actual_size:
            # convert filled_area, perimeter, equivalent diameter and feret_diameter_max for pixel value to actual
            # values
            temp_props_df['filled_area'] = temp_props_df['filled_area'].apply(self.calibrator.calibrate_filled_area)
            temp_props_df['equivalent_diameter'] = temp_props_df['equivalent_diameter'].apply(self.calibrator.calibrate_equiv_diameter)
            temp_props_df['feret_diameter_max'] = temp_props_df['feret_diameter_max'].apply(self.calibrator.calibrate_ferret_dia)
            temp_props_df['perimeter'] = temp_props_df['perimeter'].apply(self.calibrator.calibrate_perimeter)

        # compute and add 'roundness' to the dataframe
        temp_props_df['roundness'] = self._compute_roundness(area=temp_props_df['filled_area'],
                                                             perimeter=temp_props_df['perimeter'])

        # remove labeled objects with roundness > 1
        filt = (temp_props_df['roundness'] > 1)  # produce filter
        idx = temp_props_df[filt].index
        temp_props_df.drop(index=idx, inplace=True, axis=0)

        # include the Lab color values
        temp_props_df['L'], temp_props_df['a'], temp_props_df['b'] = self._produce_lab_panda_series(
            red_intensities=temp_props_df['mean_intensity-0'],
            green_intensities=temp_props_df['mean_intensity-1'],
            blue_intensities=temp_props_df['mean_intensity-2'])

        # calibrate the camera Lab values to Colorimeter values
        if self.calibrate_color:
            temp_props_df['L'] = temp_props_df['L'].apply(self.calibrator.compute_colorimeter_l_value)
            temp_props_df['a'] = temp_props_df['a'].apply(self.calibrator.compute_colorimeter_a_value)
            temp_props_df['b'] = temp_props_df['b'].apply(self.calibrator.compute_colorimeter_b_value)

        # produce a dataframe containing the textural values, then include them in the dataframe
        texture_df = self._produce_texture_dataframe(intensity_image_dict=self.temp_intensity_image_dict)
        temp_props_df.loc[::, texture_df.columns] = texture_df

        # add image_name to the dataframe
        temp_props_df.loc[::, ['image_id']] = f"img_{image_index}"

        # add the current interval
        if self.group_images_by_time:
            temp_props_df.loc[::, ['time']] = self.current_interval
            # re-arrange the columns
            temp_props_df = temp_props_df.reindex(columns=(
                    ['time', 'image_id'] + [column for column in temp_props_df.columns if
                                            column not in ['time', 'image_id']]))
        else:
            temp_props_df = temp_props_df.reindex(
                columns=(['image_id'] + [column for column in temp_props_df.columns if column != 'image_id']))

        # compute mean values of the visual properties and drop unwanted columns from the mean dataframe
        temp_mean_props_df = temp_props_df.describe().loc[['mean']].drop(
            columns=['mean_intensity-0', 'mean_intensity-1', 'mean_intensity-2', 'label'])

        temp_mean_props_df['image_id'] = f"img_{image_index}"
        # re-arrange the columns
        temp_mean_props_df = temp_mean_props_df.reindex(
            columns=(['image_id'] + [column for column in temp_mean_props_df.columns if column != 'image_id']))

        # Append the mean and comprehensive properties to their main counterparts
        # self.temp_comprehensive_props_df.append(other=temp_props_df, ignore_index=True)
        # self.temp_mean_props_df.append(other=temp_mean_props_df, ignore_index=True)
        self.temp_comprehensive_props_df = temp_props_df.copy(deep=True)
        self.temp_mean_props_df = temp_mean_props_df.copy(deep=True)

        # update the parent dataframes (comprehensive_props_df and mean_props_df)
        self._update_parent_dataframes()

        if self.realtime_update and self.save_location:
            self._save_dataframe_to_json_and_csv()

        if self.group_images_by_time:
            if counter % self.image_group_size == 0:
                self.current_interval += self.capture_interval

        return counter

    def _tf_computer_visual_properties(self, counter, image_index, rgb_image, mask):
        counter_shape = counter.shape
        [index, ] = tf.py_function(func=self._compute_visual_properties_for_single_image_and_mask,
                                   inp=[counter, image_index, rgb_image, mask],
                                   Tout=[tf.int64])
        index.set_shape(counter_shape)
        return index

    def _process_image_mask_dataset(self):
        image_mask_dataset = self.image_mask_dataset

        # create counters
        counter = tf.data.Dataset.counter(start=1)
        image_index = tf.data.Dataset.counter(start=self.image_index)

        # combine counters and data into tuple (counter, index, image, mask)
        image_mask_dataset = tf.data.Dataset.zip((counter, image_index, image_mask_dataset))
        image_mask_dataset = image_mask_dataset.map(lambda x, y, z: (x, y, z[0], z[1]))

        # set the num_parallel_calls to None, so that the images are processed and recorded
        # in the Dataframe sequentially.
        image_mask_dataset = image_mask_dataset.map(self._tf_computer_visual_properties, num_parallel_calls=None)
        self.image_mask_dataset = image_mask_dataset.prefetch(buffer_size=self.tune)

        list(image_mask_dataset.as_numpy_iterator())

    def process_data(self):
        self._process_image_mask_dataset()

        if not self.realtime_update and self.save_location:
            self._save_dataframe_to_json_and_csv()
