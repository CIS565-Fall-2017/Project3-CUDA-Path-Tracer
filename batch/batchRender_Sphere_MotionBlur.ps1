"Batch Render Start"
for($i = 0; $i -le 1; $i+= 0.025)
{
    $f = $i * 100
    .\cis565_path_tracer.exe cornell_ref1.txt $i
    "Finished: $f%"
}
"Batch Render Finished!"