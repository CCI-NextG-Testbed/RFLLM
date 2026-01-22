file = load("/home/rapcole12/Documents/RFLLM/dataset/simple/output/Case_3.mat");

%% -----------------------
% Save IQ to pred.bin
%% -----------------------
iq = file.iq;
iq = single(iq(:));   % ensure column vector, float32

fid = fopen('/home/rapcole12/Documents/RFLLM/dataset/simple/pred3.bin', 'wb');
fwrite(fid, [real(iq) imag(iq)].', 'float32');  % interleaved I Q
fclose(fid);

%% -----------------------
% Save bits to bits.bin
%% -----------------------
bits = file.bits(:);           % ensure column vector
bits = uint8(bits ~= 0);       % force to 0/1 uint8

fid = fopen('/home/rapcole12/Documents/RFLLM/dataset/simple/pred_bits3.bin', 'wb');
fwrite(fid, bits, 'uint8');
fclose(fid);

disp("Saved pred.bin (IQ) and bits.bin (uint8 bits).");
