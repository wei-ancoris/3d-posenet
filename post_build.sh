#!/bin/bash
cp mirror.png dist/mirror.png
sed -i 's/app\./tryon\/app\./g' dist/index.html
sed -i 's/\[\]/\[<\?php echo\(\$images\); \?>]/g' dist/index.html