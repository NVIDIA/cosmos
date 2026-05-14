# imagecodecs/numcodecs.py

# Copyright (c) 2021-2022, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Additional numcodecs implemented using imagecodecs."""

__version__ = "2022.9.26"

__all__ = ("register_codecs",)

import imagecodecs
import numpy
from numcodecs.abc import Codec
from numcodecs.registry import get_codec, register_codec


def protective_squeeze(x: numpy.ndarray):
    """
    Squeeze dim only if it's not the last dim.
    Image dim expected to be *, H, W, C
    """
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = numpy.prod(x.shape[:-3])
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


def get_default_image_compressor(**kwargs):
    if imagecodecs.JPEGXL:
        # has JPEGXL
        this_kwargs = {
            "effort": 3,
            "distance": 0.3,
            # bug in libjxl, invalid codestream for non-lossless
            # when decoding speed > 1
            "decodingspeed": 1,
        }
        this_kwargs.update(kwargs)
        return JpegXl(**this_kwargs)
    else:
        this_kwargs = {"level": 50}
        this_kwargs.update(kwargs)
        return Jpeg2k(**this_kwargs)


class Aec(Codec):
    """AEC codec for numcodecs."""

    codec_id = "imagecodecs_aec"

    def __init__(self, bitspersample=None, flags=None, blocksize=None, rsi=None):
        self.bitspersample = bitspersample
        self.flags = flags
        self.blocksize = blocksize
        self.rsi = rsi

    def encode(self, buf):
        return imagecodecs.aec_encode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
        )

    def decode(self, buf, out=None):
        return imagecodecs.aec_decode(
            buf,
            bitspersample=self.bitspersample,
            flags=self.flags,
            blocksize=self.blocksize,
            rsi=self.rsi,
            out=_flat(out),
        )


class Apng(Codec):
    """APNG codec for numcodecs."""

    codec_id = "imagecodecs_apng"

    def __init__(self, level=None, photometric=None, delay=None):
        self.level = level
        self.photometric = photometric
        self.delay = delay

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.apng_encode(
            buf,
            level=self.level,
            photometric=self.photometric,
            delay=self.delay,
        )

    def decode(self, buf, out=None):
        return imagecodecs.apng_decode(buf, out=out)


class Avif(Codec):
    """AVIF codec for numcodecs."""

    codec_id = "imagecodecs_avif"

    def __init__(
        self,
        level=None,
        speed=None,
        tilelog2=None,
        bitspersample=None,
        pixelformat=None,
        numthreads=None,
        index=None,
    ):
        self.level = level
        self.speed = speed
        self.tilelog2 = tilelog2
        self.bitspersample = bitspersample
        self.pixelformat = pixelformat
        self.numthreads = numthreads
        self.index = index

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.avif_encode(
            buf,
            level=self.level,
            speed=self.speed,
            tilelog2=self.tilelog2,
            bitspersample=self.bitspersample,
            pixelformat=self.pixelformat,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.avif_decode(buf, index=self.index, numthreads=self.numthreads, out=out)


class Bitorder(Codec):
    """Bitorder codec for numcodecs."""

    codec_id = "imagecodecs_bitorder"

    def encode(self, buf):
        return imagecodecs.bitorder_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.bitorder_decode(buf, out=_flat(out))


class Bitshuffle(Codec):
    """Bitshuffle codec for numcodecs."""

    codec_id = "imagecodecs_bitshuffle"

    def __init__(self, itemsize=1, blocksize=0):
        self.itemsize = itemsize
        self.blocksize = blocksize

    def encode(self, buf):
        return imagecodecs.bitshuffle_encode(buf, itemsize=self.itemsize, blocksize=self.blocksize).tobytes()

    def decode(self, buf, out=None):
        return imagecodecs.bitshuffle_decode(
            buf,
            itemsize=self.itemsize,
            blocksize=self.blocksize,
            out=_flat(out),
        )


class Blosc(Codec):
    """Blosc codec for numcodecs."""

    codec_id = "imagecodecs_blosc"

    def __init__(
        self,
        level=None,
        compressor=None,
        typesize=None,
        blocksize=None,
        shuffle=None,
        numthreads=None,
    ):
        self.level = level
        self.compressor = compressor
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = shuffle
        self.numthreads = numthreads

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.blosc_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.blosc_decode(buf, numthreads=self.numthreads, out=_flat(out))


class Blosc2(Codec):
    """Blosc2 codec for numcodecs."""

    codec_id = "imagecodecs_blosc2"

    def __init__(
        self,
        level=None,
        compressor=None,
        typesize=None,
        blocksize=None,
        shuffle=None,
        numthreads=None,
    ):
        self.level = level
        self.compressor = compressor
        self.typesize = typesize
        self.blocksize = blocksize
        self.shuffle = shuffle
        self.numthreads = numthreads

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.blosc2_encode(
            buf,
            level=self.level,
            compressor=self.compressor,
            typesize=self.typesize,
            blocksize=self.blocksize,
            shuffle=self.shuffle,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.blosc2_decode(buf, numthreads=self.numthreads, out=_flat(out))


class Brotli(Codec):
    """Brotli codec for numcodecs."""

    codec_id = "imagecodecs_brotli"

    def __init__(self, level=None, mode=None, lgwin=None):
        self.level = level
        self.mode = mode
        self.lgwin = lgwin

    def encode(self, buf):
        return imagecodecs.brotli_encode(buf, level=self.level, mode=self.mode, lgwin=self.lgwin)

    def decode(self, buf, out=None):
        return imagecodecs.brotli_decode(buf, out=_flat(out))


class ByteShuffle(Codec):
    """ByteShuffle codec for numcodecs."""

    codec_id = "imagecodecs_byteshuffle"

    def __init__(self, shape, dtype, axis=-1, dist=1, delta=False, reorder=False):
        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = axis
        self.dist = dist
        self.delta = bool(delta)
        self.reorder = bool(reorder)

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        assert buf.shape == self.shape
        assert buf.dtype == self.dtype
        return imagecodecs.byteshuffle_encode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
        ).tobytes()

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
        return imagecodecs.byteshuffle_decode(
            buf,
            axis=self.axis,
            dist=self.dist,
            delta=self.delta,
            reorder=self.reorder,
            out=out,
        )


class Bz2(Codec):
    """Bz2 codec for numcodecs."""

    codec_id = "imagecodecs_bz2"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        return imagecodecs.bz2_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.bz2_decode(buf, out=_flat(out))


class Cms(Codec):
    """CMS codec for numcodecs."""

    codec_id = "imagecodecs_cms"

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, buf, out=None):
        # return imagecodecs.cms_transform(buf)
        raise NotImplementedError

    def decode(self, buf, out=None):
        # return imagecodecs.cms_transform(buf)
        raise NotImplementedError


class Deflate(Codec):
    """Deflate codec for numcodecs."""

    codec_id = "imagecodecs_deflate"

    def __init__(self, level=None, raw=False):
        self.level = level
        self.raw = bool(raw)

    def encode(self, buf):
        return imagecodecs.deflate_encode(buf, level=self.level, raw=self.raw)

    def decode(self, buf, out=None):
        return imagecodecs.deflate_decode(buf, out=_flat(out), raw=self.raw)


class Delta(Codec):
    """Delta codec for numcodecs."""

    codec_id = "imagecodecs_delta"

    def __init__(self, shape=None, dtype=None, axis=-1, dist=1):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = axis
        self.dist = dist

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = protective_squeeze(numpy.asarray(buf))
            assert buf.shape == self.shape
            assert buf.dtype == self.dtype
        return imagecodecs.delta_encode(buf, axis=self.axis, dist=self.dist).tobytes()

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
        return imagecodecs.delta_decode(buf, axis=self.axis, dist=self.dist, out=out)


class Float24(Codec):
    """Float24 codec for numcodecs."""

    codec_id = "imagecodecs_float24"

    def __init__(self, byteorder=None, rounding=None):
        self.byteorder = byteorder
        self.rounding = rounding

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.float24_encode(buf, byteorder=self.byteorder, rounding=self.rounding)

    def decode(self, buf, out=None):
        return imagecodecs.float24_decode(buf, byteorder=self.byteorder, out=out)


class FloatPred(Codec):
    """Floating Point Predictor codec for numcodecs."""

    codec_id = "imagecodecs_floatpred"

    def __init__(self, shape, dtype, axis=-1, dist=1):
        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.axis = axis
        self.dist = dist

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        assert buf.shape == self.shape
        assert buf.dtype == self.dtype
        return imagecodecs.floatpred_encode(buf, axis=self.axis, dist=self.dist).tobytes()

    def decode(self, buf, out=None):
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
        return imagecodecs.floatpred_decode(buf, axis=self.axis, dist=self.dist, out=out)


class Gif(Codec):
    """GIF codec for numcodecs."""

    codec_id = "imagecodecs_gif"

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.gif_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.gif_decode(buf, asrgb=False, out=out)


class Heif(Codec):
    """HEIF codec for numcodecs."""

    codec_id = "imagecodecs_heif"

    def __init__(
        self,
        level=None,
        bitspersample=None,
        photometric=None,
        compression=None,
        numthreads=None,
        index=None,
    ):
        self.level = level
        self.bitspersample = bitspersample
        self.photometric = photometric
        self.compression = compression
        self.numthreads = numthreads
        self.index = index

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.heif_encode(
            buf,
            level=self.level,
            bitspersample=self.bitspersample,
            photometric=self.photometric,
            compression=self.compression,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.heif_decode(
            buf,
            index=self.index,
            photometric=self.photometric,
            numthreads=self.numthreads,
            out=out,
        )


class Jetraw(Codec):
    """Jetraw codec for numcodecs."""

    codec_id = "imagecodecs_jetraw"

    def __init__(
        self,
        shape,
        identifier,
        parameters=None,
        verbosity=None,
        errorbound=None,
    ):
        self.shape = shape
        self.identifier = identifier
        self.errorbound = errorbound
        imagecodecs.jetraw_init(parameters, verbosity)

    def encode(self, buf):
        return imagecodecs.jetraw_encode(buf, identifier=self.identifier, errorbound=self.errorbound)

    def decode(self, buf, out=None):
        if out is None:
            out = numpy.empty(self.shape, numpy.uint16)
        return imagecodecs.jetraw_decode(buf, out=out)


class Jpeg(Codec):
    """JPEG codec for numcodecs."""

    codec_id = "imagecodecs_jpeg"

    def __init__(
        self,
        bitspersample=None,
        tables=None,
        header=None,
        colorspace_data=None,
        colorspace_jpeg=None,
        level=None,
        subsampling=None,
        optimize=None,
        smoothing=None,
    ):
        self.tables = tables
        self.header = header
        self.bitspersample = bitspersample
        self.colorspace_data = colorspace_data
        self.colorspace_jpeg = colorspace_jpeg
        self.level = level
        self.subsampling = subsampling
        self.optimize = optimize
        self.smoothing = smoothing

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpeg_encode(
            buf,
            level=self.level,
            colorspace=self.colorspace_data,
            outcolorspace=self.colorspace_jpeg,
            subsampling=self.subsampling,
            optimize=self.optimize,
            smoothing=self.smoothing,
        )

    def decode(self, buf, out=None):
        out_shape = None
        if out is not None:
            out_shape = out.shape
            out = protective_squeeze(out)
        img = imagecodecs.jpeg_decode(
            buf,
            bitspersample=self.bitspersample,
            tables=self.tables,
            header=self.header,
            colorspace=self.colorspace_jpeg,
            outcolorspace=self.colorspace_data,
            out=out,
        )
        if out_shape is not None:
            img = img.reshape(out_shape)
        return img

    def get_config(self):
        """Return dictionary holding configuration parameters."""
        config = dict(id=self.codec_id)
        for key in self.__dict__:
            if not key.startswith("_"):
                value = getattr(self, key)
                if value is not None and key in ("header", "tables"):
                    import base64

                    value = base64.b64encode(value).decode()
                config[key] = value
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiate codec from configuration object."""
        for key in ("header", "tables"):
            value = config.get(key, None)
            if value is not None and isinstance(value, str):
                import base64

                config[key] = base64.b64decode(value.encode())
        return cls(**config)


class Jpeg2k(Codec):
    """JPEG 2000 codec for numcodecs."""

    codec_id = "imagecodecs_jpeg2k"

    def __init__(
        self,
        level=None,
        codecformat=None,
        colorspace=None,
        tile=None,
        reversible=None,
        bitspersample=None,
        resolutions=None,
        numthreads=None,
        verbose=0,
    ):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpeg2k_encode(
            buf,
            level=self.level,
            codecformat=self.codecformat,
            colorspace=self.colorspace,
            tile=self.tile,
            reversible=self.reversible,
            bitspersample=self.bitspersample,
            resolutions=self.resolutions,
            numthreads=self.numthreads,
            verbose=self.verbose,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpeg2k_decode(buf, verbose=self.verbose, numthreads=self.numthreads, out=out)


class JpegLs(Codec):
    """JPEG LS codec for numcodecs."""

    codec_id = "imagecodecs_jpegls"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpegls_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.jpegls_decode(buf, out=out)


class JpegXl(Codec):
    """JPEG XL codec for numcodecs."""

    codec_id = "imagecodecs_jpegxl"

    def __init__(
        self,
        # encode
        level=None,
        effort=None,
        distance=None,
        lossless=None,
        decodingspeed=None,
        photometric=None,
        planar=None,
        usecontainer=None,
        # decode
        index=None,
        keeporientation=None,
        # both
        numthreads=None,
    ):
        """
        Return JPEG XL image from numpy array.
        Float must be in nominal range 0..1.

        Currently L, LA, RGB, RGBA images are supported in contig mode.
        Extra channels are only supported for grayscale images in planar mode.

        Parameters
        ----------
        level : Default to None, i.e. not overwriting lossess and decodingspeed options.
            When < 0: Use lossless compression
            When in [0,1,2,3,4]: Sets the decoding speed tier for the provided options.
                Minimum is 0 (slowest to decode, best quality/density), and maximum
                is 4 (fastest to decode, at the cost of some quality/density).
        effort : Default to 3.
            Sets encoder effort/speed level without affecting decoding speed.
            Valid values are, from faster to slower speed: 1:lightning 2:thunder
                3:falcon 4:cheetah 5:hare 6:wombat 7:squirrel 8:kitten 9:tortoise.
            Speed: lightning, thunder, falcon, cheetah, hare, wombat, squirrel, kitten, tortoise
            control the encoder effort in ascending order.
            This also affects memory usage: using lower effort will typically reduce memory
            consumption during encoding.
            lightning and thunder are fast modes useful for lossless mode (modular).
            falcon disables all of the following tools.
            cheetah enables coefficient reordering, context clustering, and heuristics for selecting DCT sizes and quantization steps.
            hare enables Gaborish filtering, chroma from luma, and an initial estimate of quantization steps.
            wombat enables error diffusion quantization and full DCT size selection heuristics.
            squirrel (default) enables dots, patches, and spline detection, and full context clustering.
            kitten optimizes the adaptive quantization for a psychovisual metric.
            tortoise enables a more thorough adaptive quantization search.
        distance : Default to 1.0
            Sets the distance level for lossy compression: target max butteraugli distance,
            lower = higher quality. Range: 0 .. 15. 0.0 = mathematically lossless
            (however, use JxlEncoderSetFrameLossless instead to use true lossless,
            as setting distance to 0 alone is not the only requirement).
            1.0 = visually lossless. Recommended range: 0.5 .. 3.0.
        lossess : Default to False.
            Use lossess encoding.
        decodingspeed : Default to 0.
            Duplicate to level. [0,4]
        photometric : Return JxlColorSpace value.
            Default logic is quite complicated but works most of the time.
            Accepted value:
                int: [-1,3]
                str: ['RGB',
                    'WHITEISZERO', 'MINISWHITE',
                    'BLACKISZERO', 'MINISBLACK', 'GRAY',
                    'XYB', 'KNOWN']
        planar : Enable multi-channel mode.
            Default to false.
        usecontainer :
            Forces the encoder to use the box-based container format (BMFF)
            even when not necessary.
            When using JxlEncoderUseBoxes, JxlEncoderStoreJPEGMetadata or
            JxlEncoderSetCodestreamLevel with level 10, the encoder will
            automatically also use the container format, it is not necessary
            to use JxlEncoderUseContainer for those use cases.
            By default this setting is disabled.
        index : Selectively decode frames for animation.
            Default to 0, decode all frames.
            When set to > 0, decode that frame index only.
        keeporientation :
            Enables or disables preserving of as-in-bitstream pixeldata orientation.
            Some images are encoded with an Orientation tag indicating that the
            decoder must perform a rotation and/or mirroring to the encoded image data.

            If skip_reorientation is JXL_FALSE (the default): the decoder will apply
            the transformation from the orientation setting, hence rendering the image
            according to its specified intent. When producing a JxlBasicInfo, the decoder
            will always set the orientation field to JXL_ORIENT_IDENTITY (matching the
            returned pixel data) and also align xsize and ysize so that they correspond
            to the width and the height of the returned pixel data.

            If skip_reorientation is JXL_TRUE: the decoder will skip applying the
            transformation from the orientation setting, returning the image in
            the as-in-bitstream pixeldata orientation. This may be faster to decode
            since the decoder doesnt have to apply the transformation, but can
            cause wrong display of the image if the orientation tag is not correctly
            taken into account by the user.

            By default, this option is disabled, and the returned pixel data is
            re-oriented according to the images Orientation setting.
        threads : Default to 1.
            If <= 0, use all cores.
            If > 32, clipped to 32.
        """

        self.level = level
        self.effort = effort
        self.distance = distance
        self.lossless = bool(lossless)
        self.decodingspeed = decodingspeed
        self.photometric = photometric
        self.planar = planar
        self.usecontainer = usecontainer
        self.index = index
        self.keeporientation = keeporientation
        self.numthreads = numthreads

    def encode(self, buf):

        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpegxl_encode(
            buf,
            level=self.level,
            effort=self.effort,
            distance=self.distance,
            lossless=self.lossless,
            decodingspeed=self.decodingspeed,
            photometric=self.photometric,
            planar=self.planar,
            usecontainer=self.usecontainer,
            numthreads=self.numthreads,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpegxl_decode(
            buf,
            index=self.index,
            keeporientation=self.keeporientation,
            numthreads=self.numthreads,
            out=out,
        )


class JpegXr(Codec):
    """JPEG XR codec for numcodecs."""

    codec_id = "imagecodecs_jpegxr"

    def __init__(
        self,
        level=None,
        photometric=None,
        hasalpha=None,
        resolution=None,
        fp2int=None,
    ):
        self.level = level
        self.photometric = photometric
        self.hasalpha = hasalpha
        self.resolution = resolution
        self.fp2int = fp2int

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.jpegxr_encode(
            buf,
            level=self.level,
            photometric=self.photometric,
            hasalpha=self.hasalpha,
            resolution=self.resolution,
        )

    def decode(self, buf, out=None):
        return imagecodecs.jpegxr_decode(buf, fp2int=self.fp2int, out=out)


class Lerc(Codec):
    """LERC codec for numcodecs."""

    codec_id = "imagecodecs_lerc"

    def __init__(self, level=None, version=None, planar=None):
        self.level = level
        self.version = version
        self.planar = bool(planar)

        # self.mask = None

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.lerc_encode(
            buf,
            level=self.level,
            version=self.version,
            planar=self.planar,
        )

    def decode(self, buf, out=None):
        return imagecodecs.lerc_decode(buf, out=out)


class Ljpeg(Codec):
    """LJPEG codec for numcodecs."""

    codec_id = "imagecodecs_ljpeg"

    def __init__(self, bitspersample=None):
        self.bitspersample = bitspersample

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.ljpeg_encode(buf, bitspersample=self.bitspersample)

    def decode(self, buf, out=None):
        return imagecodecs.ljpeg_decode(buf, out=out)


class Lz4(Codec):
    """LZ4 codec for numcodecs."""

    codec_id = "imagecodecs_lz4"

    def __init__(self, level=None, hc=False, header=True):
        self.level = level
        self.hc = hc
        self.header = bool(header)

    def encode(self, buf):
        return imagecodecs.lz4_encode(buf, level=self.level, hc=self.hc, header=self.header)

    def decode(self, buf, out=None):
        return imagecodecs.lz4_decode(buf, header=self.header, out=_flat(out))


class Lz4f(Codec):
    """LZ4F codec for numcodecs."""

    codec_id = "imagecodecs_lz4f"

    def __init__(
        self,
        level=None,
        blocksizeid=False,
        contentchecksum=None,
        blockchecksum=None,
    ):
        self.level = level
        self.blocksizeid = blocksizeid
        self.contentchecksum = contentchecksum
        self.blockchecksum = blockchecksum

    def encode(self, buf):
        return imagecodecs.lz4f_encode(
            buf,
            level=self.level,
            blocksizeid=self.blocksizeid,
            contentchecksum=self.contentchecksum,
            blockchecksum=self.blockchecksum,
        )

    def decode(self, buf, out=None):
        return imagecodecs.lz4f_decode(buf, out=_flat(out))


class Lzf(Codec):
    """LZF codec for numcodecs."""

    codec_id = "imagecodecs_lzf"

    def __init__(self, header=True):
        self.header = bool(header)

    def encode(self, buf):
        return imagecodecs.lzf_encode(buf, header=self.header)

    def decode(self, buf, out=None):
        return imagecodecs.lzf_decode(buf, header=self.header, out=_flat(out))


class Lzma(Codec):
    """LZMA codec for numcodecs."""

    codec_id = "imagecodecs_lzma"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        return imagecodecs.lzma_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.lzma_decode(buf, out=_flat(out))


class Lzw(Codec):
    """LZW codec for numcodecs."""

    codec_id = "imagecodecs_lzw"

    def encode(self, buf):
        return imagecodecs.lzw_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.lzw_decode(buf, out=_flat(out))


class PackBits(Codec):
    """PackBits codec for numcodecs."""

    codec_id = "imagecodecs_packbits"

    def __init__(self, axis=None):
        self.axis = axis

    def encode(self, buf):
        if not isinstance(buf, (bytes, bytearray)):
            buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.packbits_encode(buf, axis=self.axis)

    def decode(self, buf, out=None):
        return imagecodecs.packbits_decode(buf, out=_flat(out))


class Pglz(Codec):
    """PGLZ codec for numcodecs."""

    codec_id = "imagecodecs_pglz"

    def __init__(self, header=True, strategy=None):
        self.header = bool(header)
        self.strategy = strategy

    def encode(self, buf):
        return imagecodecs.pglz_encode(buf, strategy=self.strategy, header=self.header)

    def decode(self, buf, out=None):
        return imagecodecs.pglz_decode(buf, header=self.header, out=_flat(out))


class Png(Codec):
    """PNG codec for numcodecs."""

    codec_id = "imagecodecs_png"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.png_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.png_decode(buf, out=out)


class Qoi(Codec):
    """QOI codec for numcodecs."""

    codec_id = "imagecodecs_qoi"

    def __init__(self):
        pass

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.qoi_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.qoi_decode(buf, out=out)


class Rgbe(Codec):
    """RGBE codec for numcodecs."""

    codec_id = "imagecodecs_rgbe"

    def __init__(self, header=False, shape=None, rle=None):
        if not header and shape is None:
            raise ValueError("must specify data shape if no header")
        if shape and shape[-1] != 3:
            raise ValueError("invalid shape")
        self.shape = shape
        self.header = bool(header)
        self.rle = None if rle is None else bool(rle)

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.rgbe_encode(buf, header=self.header, rle=self.rle)

    def decode(self, buf, out=None):
        if out is None and not self.header:
            out = numpy.empty(self.shape, numpy.float32)
        return imagecodecs.rgbe_decode(buf, header=self.header, rle=self.rle, out=out)


class Rcomp(Codec):
    """Rcomp codec for numcodecs."""

    codec_id = "imagecodecs_rcomp"

    def __init__(self, shape, dtype, nblock=None):
        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype).str
        self.nblock = nblock

    def encode(self, buf):
        return imagecodecs.rcomp_encode(buf, nblock=self.nblock)

    def decode(self, buf, out=None):
        return imagecodecs.rcomp_decode(
            buf,
            shape=self.shape,
            dtype=self.dtype,
            nblock=self.nblock,
            out=out,
        )


class Snappy(Codec):
    """Snappy codec for numcodecs."""

    codec_id = "imagecodecs_snappy"

    def encode(self, buf):
        return imagecodecs.snappy_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.snappy_decode(buf, out=_flat(out))


class Spng(Codec):
    """SPNG codec for numcodecs."""

    codec_id = "imagecodecs_spng"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.spng_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.spng_decode(buf, out=out)


class Tiff(Codec):
    """TIFF codec for numcodecs."""

    codec_id = "imagecodecs_tiff"

    def __init__(self, index=None, asrgb=None, verbose=0):
        self.index = index
        self.asrgb = bool(asrgb)
        self.verbose = verbose

    def encode(self, buf):

        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.tiff_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.tiff_decode(
            buf,
            index=self.index,
            asrgb=self.asrgb,
            verbose=self.verbose,
            out=out,
        )


class Webp(Codec):
    """WebP codec for numcodecs."""

    codec_id = "imagecodecs_webp"

    def __init__(self, level=None, lossless=None, method=None, hasalpha=None):
        self.level = level
        self.hasalpha = bool(hasalpha)
        self.method = method
        self.lossless = lossless

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        return imagecodecs.webp_encode(buf, level=self.level, lossless=self.lossless, method=self.method)

    def decode(self, buf, out=None):
        return imagecodecs.webp_decode(buf, hasalpha=self.hasalpha, out=out)


class Xor(Codec):
    """XOR codec for numcodecs."""

    codec_id = "imagecodecs_xor"

    def __init__(self, shape=None, dtype=None, axis=-1):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else numpy.dtype(dtype).str
        self.axis = axis

    def encode(self, buf):
        if self.shape is not None or self.dtype is not None:
            buf = protective_squeeze(numpy.asarray(buf))
            assert buf.shape == self.shape
            assert buf.dtype == self.dtype
        return imagecodecs.xor_encode(buf, axis=self.axis).tobytes()

    def decode(self, buf, out=None):
        if self.shape is not None or self.dtype is not None:
            buf = numpy.frombuffer(buf, dtype=self.dtype).reshape(*self.shape)
        return imagecodecs.xor_decode(buf, axis=self.axis, out=_flat(out))


class Zfp(Codec):
    """ZFP codec for numcodecs."""

    codec_id = "imagecodecs_zfp"

    def __init__(
        self,
        shape=None,
        dtype=None,
        strides=None,
        level=None,
        mode=None,
        execution=None,
        numthreads=None,
        chunksize=None,
        header=True,
    ):
        if header:
            self.shape = None
            self.dtype = None
            self.strides = None
        elif shape is None or dtype is None:
            raise ValueError("invalid shape or dtype")
        else:
            self.shape = tuple(shape)
            self.dtype = numpy.dtype(dtype).str
            self.strides = None if strides is None else tuple(strides)
        self.level = level
        self.mode = mode
        self.execution = execution
        self.numthreads = numthreads
        self.chunksize = chunksize
        self.header = bool(header)

    def encode(self, buf):
        buf = protective_squeeze(numpy.asarray(buf))
        if not self.header:
            assert buf.shape == self.shape
            assert buf.dtype == self.dtype
        return imagecodecs.zfp_encode(
            buf,
            level=self.level,
            mode=self.mode,
            execution=self.execution,
            header=self.header,
            numthreads=self.numthreads,
            chunksize=self.chunksize,
        )

    def decode(self, buf, out=None):
        if self.header:
            return imagecodecs.zfp_decode(buf, out=out)
        return imagecodecs.zfp_decode(
            buf,
            shape=self.shape,
            dtype=numpy.dtype(self.dtype),
            strides=self.strides,
            numthreads=self.numthreads,
            out=out,
        )


class Zlib(Codec):
    """Zlib codec for numcodecs."""

    codec_id = "imagecodecs_zlib"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        return imagecodecs.zlib_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zlib_decode(buf, out=_flat(out))


class Zlibng(Codec):
    """Zlibng codec for numcodecs."""

    codec_id = "imagecodecs_zlibng"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        return imagecodecs.zlibng_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zlibng_decode(buf, out=_flat(out))


class Zopfli(Codec):
    """Zopfli codec for numcodecs."""

    codec_id = "imagecodecs_zopfli"

    def encode(self, buf):
        return imagecodecs.zopfli_encode(buf)

    def decode(self, buf, out=None):
        return imagecodecs.zopfli_decode(buf, out=_flat(out))


class Zstd(Codec):
    """ZStandard codec for numcodecs."""

    codec_id = "imagecodecs_zstd"

    def __init__(self, level=None):
        self.level = level

    def encode(self, buf):
        return imagecodecs.zstd_encode(buf, level=self.level)

    def decode(self, buf, out=None):
        return imagecodecs.zstd_decode(buf, out=_flat(out))


def _flat(out):
    """Return numpy array as contiguous view of bytes if possible."""
    if out is None:
        return None
    view = memoryview(out)
    if view.readonly or not view.contiguous:
        return None
    return view.cast("B")


def register_codecs(codecs=None, force=False, verbose=True):
    """Register codecs in this module with numcodecs."""
    for name, cls in globals().items():
        if not hasattr(cls, "codec_id") or name == "Codec":
            continue
        if codecs is not None and cls.codec_id not in codecs:
            continue
        try:
            try:
                get_codec({"id": cls.codec_id})
            except TypeError:
                # registered, but failed
                pass
        except ValueError:
            # not registered yet
            pass
        else:
            if not force:
                if verbose:
                    log_warning(f"numcodec {cls.codec_id!r} already registered")
                continue
            if verbose:
                log_warning(f"replacing registered numcodec {cls.codec_id!r}")
        register_codec(cls)


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)
