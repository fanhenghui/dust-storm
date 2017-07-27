#version 430

bool check_opacity(in out float opacity)
{
    if(opacity > 0.95)
    {
        opacity = 1.0;
        return true;
    }
    else
    {
        return false;
    }
}

//Encoding label to intger array 4*32 can contain 0~127 labels
void label_encode(int label , in out int mask_flag[4])
{
    if(label < 32)
    {
        mask_flag[0] = mask_flag[0] | ( 1 << label );
    }
    else if(label < 64)
    {
        mask_flag[1] = mask_flag[1] | ( 1 << (label-32) );
    }
    else if(label < 96)
    {
        mask_flag[2] = mask_flag[2] | ( 1 << (label-64) );
    }
    else
    {
        mask_flag[3] = mask_flag[3] | ( 1 << (label-96) );
    }
}

//Decoding label from intger array 4*32 can contain 0~127 labels
bool label_decode(int label , int mask_flag[4])
{

    bool is_hitted = false;
    if(label < 32)
    {
        is_hitted = ( ( 1 << label ) & mask_flag[0] ) != 0;
    }
    else if(label < 64)
    {
        is_hitted = ( ( 1 << (label - 32) ) & mask_flag[1] ) != 0;
    }
    else if(label < 96)
    {
        is_hitted = ( ( 1 << (label - 64) ) & mask_flag[2] ) != 0;
    }
    else
    {
        is_hitted = ( ( 1 << (label - 96) ) & mask_flag[3] ) != 0;
    }
    return is_hitted;
}