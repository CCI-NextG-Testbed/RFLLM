for k = 0:3
    % -----------------------
    % Load Case_k.mat
    % -----------------------
caseFile = sprintf('/home/rapcole12/Documents/RFLLM/results/prediction_batch/pred_test_000%d.mat', k);
    file = load(caseFile);

    % -----------------------
    % Save IQ to pred{k}.bin
    % -----------------------
    iq = file.iq;
    iq = single(iq(:));   % ensure column vector, float32

    predIQ = sprintf('/home/rapcole12/Documents/RFLLM/dataset/simple/pred_000%d.bin', k);
    fid = fopen(predIQ, 'wb');
    fwrite(fid, [real(iq) imag(iq)].', 'float32');  % interleaved I Q
    fclose(fid);

    % -----------------------
    % Save bits to pred_bits{k}.bin
    % -----------------------
    bits = file.bits(:);        % ensure column vector
    bits = uint8(bits ~= 0);    % force to 0/1 uint8

    predBits = sprintf('/home/rapcole12/Documents/RFLLM/dataset/simple/pred_bits_000%d.bin', k);
    fid = fopen(predBits, 'wb');
    fwrite(fid, bits, 'uint8');
    fclose(fid);

    fprintf("Saved pred%d.bin and pred_bits%d.bin\n", k, k);
end
