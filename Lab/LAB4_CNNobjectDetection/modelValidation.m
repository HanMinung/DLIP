clear

groundTruth = load('LAB_Parking_counting_result_answer_student_modified.txt');
model = load('counting_result.txt');
error = 0;

for Idx = 1 : 1501
    
    if groundTruth(Idx,2) ~= model(Idx,2)
    
        error = error + 1;
    end

end

numFrames = [error; 1500 - error; (1500 - error)/1500];

T = table(numFrames, 'RowNames', {'model ~= truth', 'model == truth', 'accuracy [%]'});



