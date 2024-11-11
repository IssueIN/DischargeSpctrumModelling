from spectrometer.elements import PlanarReflection, PlanarDiffractionGrating, OutputPlane

class Spectrometer:
    def __init__(self, Mirror1, DiffGrat, Mirror2, Opp):
        if not isinstance(Mirror1, PlanarReflection):
            raise TypeError("Mirror1 must be an instance of the PlanarReflection class.")
        if not isinstance(Mirror2, PlanarReflection):
            raise TypeError("Mirror2 must be an instance of the PlanarReflection class.")
        if not isinstance(DiffGrat, PlanarDiffractionGrating):
            raise TypeError("DiffGrat must be an instance of the PlanarDiffractionGrating class.")
        if not isinstance(Opp, OutputPlane):
            raise TypeError('Opp Mmust be an instance of the OutputPlane')
        
        self.__setup = [Mirror1, DiffGrat, Mirror2, Opp]
    
    def setup(self):
        return self.__setup

    def propagate(self, ray, m = None):
        for i, optics in enumerate(self.setup()):
            if i == 1 and m is not None:
                optics.propagate_ray(ray, m)
            else:
                optics.propagate_ray(ray)


# Imaging Spectrometer: the grating equation still applies to the grating surface.
        