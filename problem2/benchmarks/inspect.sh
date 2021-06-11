ho "Data format:"
echo "  <int:number of threads>"
echo "  <double:total time>"
echo ""
echo "p    num_threads     total_time"
hexdump -v -e '"%i     %f s "' -e '"\n"' $1
