#!/bin/sh
# A short script to insert a blank page every other page of an A4 pdf
# Primarily this is for printing music when only double sided is available
convert xc:none -page 595x842 blank.pdf
echo "Enter file name"
read filename
pages="`pdftk $filename dump_data | grep NumberOfPages | cut -d : -f2`"
numpages=`for ((a=1; a <= $pages; a++)); do echo -n "A$a B1 "; done`
pdftk A=$filename B=blank.pdf cat $numpages output $filename-singlesided.pdf
rm blank.pdf
exit 0
