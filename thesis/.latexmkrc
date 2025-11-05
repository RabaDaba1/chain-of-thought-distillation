@default_files = ('main.tex');

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

$biber = 'biber %O %B';
$max_repeat = 5;

@generated_exts = (@generated_exts, 'run.xml', 'bcf', 'bbl', 'blg', 'synctex.gz');