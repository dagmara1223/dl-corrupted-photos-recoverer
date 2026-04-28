Features so far:
 - Finds JPEGs
 - Validates headers and rejects fake headers - (JFIF/EXIF)
 - Discards image if APP metadata not present (APP0/APP1)
 - Limits size, in case no end marker is found - currently 20 mb
 - Produces output files