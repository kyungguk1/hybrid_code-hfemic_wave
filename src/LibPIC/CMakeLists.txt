
add_library(LibPIC EXCLUDE_FROM_ALL)
#add_library(LibPIC)
target_sources(LibPIC PUBLIC
    PIC/Config.h
    PIC/Predefined.h
    PIC/Misc/RootFinder.h
    PIC/Misc/Faddeeva.hh
    PIC/Misc/Faddeeva.cc
    PIC/UTL/println.h
    PIC/UTL/lippincott.h
    PIC/UTL/lippincott.cc
    PIC/UTL/Badge.h
    PIC/UTL/Defer.h
    PIC/UTL/Options.h
    PIC/UTL/Options.cc
    PIC/UTL/Range.h
    PIC/Random/BitReversedPattern.h
    PIC/Random/splitmix64.h
    PIC/Random/xoroshiro128.h
    PIC/Random/NRRandomEngine.h
    PIC/RandomReal.h
    PIC/RandomReal.cc
    PIC/MaskingFunction.h
    PIC/VT/Scalar.h
    PIC/VT/VectorTemplate.h
    PIC/VT/Vector.h
    PIC/VT/ComplexVector.h
    PIC/VT/TensorTemplate.h
    PIC/VT/Tensor.h
    PIC/VT/Matrix.h
    PIC/VT/FourVectorTemplate.h
    PIC/VT/FourVector.h
    PIC/VT/FourTensorTemplate.h
    PIC/VT/FourTensor.h
    PIC/Shape.h
    PIC/GridArray.h
    PIC/BorisPush.h
    PIC/CartCoord.h
    PIC/CurviCoord.h
    PIC/Particle.h
    PIC/RelativisticParticle.h
    PIC/TypeMaps.h
    PIC/PlasmaDesc.h
    PIC/Geometry/CurviBasis.h
    PIC/Geometry/MFABasis.h
    PIC/Geometry/MirrorGeometry.h
    PIC/Geometry/MirrorGeometry.cc
    PIC/Geometry.h
    PIC/Geometry.cc
    PIC/VDFHelper.h
    PIC/VDF.h
    PIC/VDF/TestParticleVDF.h
    PIC/VDF/TestParticleVDF.cc
    PIC/VDF/MaxwellianVDF.h
    PIC/VDF/MaxwellianVDF.cc
    PIC/VDF/LossconeVDF.h
    PIC/VDF/LossconeVDF.cc
    PIC/VDF/PartialShellVDF.h
    PIC/VDF/PartialShellVDF.cc
    PIC/VDF/CounterBeamVDF.h
    PIC/VDF/CounterBeamVDF.cc
    PIC/VDFVariant.h
    PIC/RelativisticVDF.h
    PIC/RelativisticVDF/TestParticleVDF.h
    PIC/RelativisticVDF/TestParticleVDF.cc
    PIC/RelativisticVDF/MaxwellianVDF.h
    PIC/RelativisticVDF/MaxwellianVDF.cc
    PIC/RelativisticVDF/LossconeVDF.h
    PIC/RelativisticVDF/LossconeVDF.cc
    PIC/RelativisticVDF/PartialShellVDF.h
    PIC/RelativisticVDF/PartialShellVDF.cc
    PIC/RelativisticVDF/CounterBeamVDF.h
    PIC/RelativisticVDF/CounterBeamVDF.cc
    PIC/RelativisticVDFVariant.h
    )

set_target_properties(LibPIC PROPERTIES
    LINKER_LANGUAGE CXX
    OUTPUT_NAME PIC
    )
target_include_directories(LibPIC PUBLIC .)

set_project_warnings(LibPIC)
enable_sanitizers(LibPIC)

add_subdirectory(test)
