# RVMEP
Right Ventricle MEsh Processing (RVMEP) tools for the cardiac right ventricle. The methods were implemented in Python during my PhD, and published in [1]. So far, the methods work only with 3D surface meshes generated using Tomtec 4D RV-FUNCTION software, but can be easily adapted for other formats. The methods included are:

- Generation of longitudinal and circumferential directions following [2]
- Computation of regional volumes (apical, basal, outlet) following the method descrived in [1]. An javascript version of this to test without installation is available at https://gbernardino.github.io/RVparcellation/#/computation. It is an static webpage and therefore there is no data transfer to the server.
- Synthetic deformation of a mesh, by prescribing strains in longitudinal/circumferential directions.
- Strain computation in basis.
- Mesh reconstruction (work in progress)

## Contact
Please send me an email at gabriel.bernardino@upf.edu

## Contributors
- Gabriel Bernardino
- Thomas Dargent (strain)

## References

[1] Bernardino G, Hodzic A, Langet H, Legallois D, De Craene M, González Ballester MÁ, Saloux É, Bijnens B. Volumetric parcellation of the cardiac right ventricle for regional geometric and functional assessment. Med Image Anal. 2021 Jul;71:102044. doi: 10.1016/j.media.2021.102044. Epub 2021 Apr 6. PMID: 33872960.

[2] Doste R, Soto-Iglesias D, Bernardino G, Alcaine A, Sebastian R, Giffard-Roisin S, Sermesant M, Berruezo A, Sanchez-Quintana D, Camara O. A rule-based method to model myocardial fiber orientation in cardiac biventricular geometries with outflow tracts. Int J Numer Method Biomed Eng. 2019 Apr;35(4):e3185. doi: 10.1002/cnm.3185. Epub 2019 Feb 19. PMID: 30721579.

[3] Bernardino, G., Dargent, T., Camara, O., Duchateau, N. (2023). Strainger Things: Discrete Differential Geometry for Transporting Right Ventricular Deformation Across Meshes. In: Bernard, O., Clarysse, P., Duchateau, N., Ohayon, J., Viallon, M. (eds) Functional Imaging and Modeling of the Heart. FIMH 2023. Lecture Notes in Computer Science, vol 13958. Springer, Cham. https://doi.org/10.1007/978-3-031-35302-4_35
