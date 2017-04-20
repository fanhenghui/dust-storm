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
void label_encode(int iLabel , in out int maskFlag[4])
{
    if(iLabel < 32)
    {
        maskFlag[0] = maskFlag[0] | ( 1 << iLabel );
    }
    else if(iLabel < 64)
    {
        maskFlag[1] = maskFlag[1] | ( 1 << (iLabel-32) );
    }
    else if(iLabel < 96)
    {
        maskFlag[2] = maskFlag[2] | ( 1 << (iLabel-64) );
    }
    else
    {
        maskFlag[3] = maskFlag[3] | ( 1 << (iLabel-96) );
    }
}

//Decoding label from intger array 4*32 can contain 0~127 labels
bool label_decode(int iLabel , int maskFlag[4])
{

    bool bHitted = false;
    if(iLabel < 32)
    {
        bHitted = ( ( 1 << iLabel ) & maskFlag[0] ) != 0;
    }
    else if(iLabel < 64)
    {
        bHitted = ( ( 1 << (iLabel - 32) ) & maskFlag[1] ) != 0;
    }
    else if(iLabel < 96)
    {
        bHitted = ( ( 1 << (iLabel - 64) ) & maskFlag[2] ) != 0;
    }
    else
    {
        bHitted = ( ( 1 << (iLabel - 96) ) & maskFlag[3] ) != 0;
    }
    return bHitted;
}