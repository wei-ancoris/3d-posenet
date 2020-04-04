#!/bin/bash
cp mirror.png dist/mirror.png
sed -i 's/app\./tryon\/app\./g' dist/index.html
#sed -i 's/\[\]/\[<\?php echo\(\$images\); \?>]/g' dist/index.html
#sudo rm -Rf /var/www/isabellazhang/tryon
#sudo cp -R dist /var/www/isabellazhang/tryon
#sudo chown -R www-data:www-data /var/www/isabellazhang/tryon