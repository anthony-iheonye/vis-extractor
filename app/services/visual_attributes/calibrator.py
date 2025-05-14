

class VisualPropertiesCalibrator:
    """Calibrates LAB* color values from camera values to colorimeter values.

    The color values were extracted from images captured offline, or in real-time during fluidized bed drying """

    def __init__(self, realtime_capture_mode: bool = True):
        """
        Calibrates LAB* color values from camera values to colorimeter values.

        :param realtime_capture_mode: Set to True, if the color values were extracted from real-time image captured during fluidized bed drying.
        """
        self.realtime_capture_mode = realtime_capture_mode

        if self.realtime_capture_mode:
            self.L_index = {'a': -5.04461047576763E+06,
                            'b': 5.13323641056063E+05,
                            'c': 1.15843677123398E+04,
                            'd': -7.50471807730221E+01}

            self.a_index = {'a': -4.37022874993402E+00,
                            'b': 7.66127417312916E-01,
                            'c': 5.72699875489546E-03,
                            'd': -1.01255846619522E-04}

            self.b_index = {'a': -1.15950594393709E+01,
                            'b': 7.12135024826324E-01,
                            'c': -2.46027845221479E-03,
                            'd': -2.19862222769195E-05}
        else:

            self.L_index = {'a': -7.62575299409734E+06,
                            'b': 2.01439605860798E+05,
                            'c': 3.65654002601337E+03,
                            'd': -2.27819646080186E+01}

            self.a_index = {'a': 2.33420333239659E+00,
                            'b': 7.18504904657152E-01,
                            'c': -3.63317332375604E-03,
                            'd': -1.01159962014828E-03}

            self.b_index = {'a': -8.65155804603329E+00,
                            'b': 8.60459574288960E-01,
                            'c': -3.59994003312128E-03,
                            'd': 6.64414157033416E-05}


    @staticmethod
    def colorimeter_value(x: float, a: float, b: float, c: float, d: float):
        """Compute the colorimeter equivalent for one of the values of the camera LAB values.
         The output is computed using a Rational model whose coefficients are a, b, c, and d."""
        output = (a + (b * x)) / (1 + (c * x) + (d * x ** 2))
        return output



    def compute_colorimeter_l_value(self, camera_lab_l_value: float):
        """
        Returns the colorimeter equivalent for L, the lightness-index.

        :param camera_lab_l_value: (float) The Camera's Lab color space l (lightness) value.
        :return: (float) The colorimeter equivalence of the lab index value
        """
        return self.colorimeter_value(x=camera_lab_l_value,
                                      a=self.L_index['a'],
                                      b=self.L_index['b'],
                                      c=self.L_index['c'],
                                      d=self.L_index['d'])

    def compute_colorimeter_a_value(self, camera_lab_a_value: float):
        """
        Returns the colorimeter equivalent for a-index.

        :param camera_lab_a_value: (float) the a-index (greenish/redness) for the camera.
        :return: (float) The colorimeter equivalence of the camera a-index.
        """
        return self.colorimeter_value(x=camera_lab_a_value,
                                      a=self.a_index['a'],
                                      b=self.a_index['b'],
                                      c=self.a_index['c'],
                                      d=self.a_index['d'])

    def compute_colorimeter_b_value(self, camera_lab_b_value: float):
        """
        Returns the colorimeter equivalent for b-index.

        :param camera_lab_b_value: (float) the b-index (greenish/redness) for the camera.
        :return: (float) The colorimeter equivalence of the camera b-index.
        """
        return self.colorimeter_value(x=camera_lab_b_value,
                                      a=self.b_index['a'],
                                      b=self.b_index['b'],
                                      c=self.b_index['c'],
                                      d=self.b_index['d'])
