#include <iostream>
#include "MedImgArithmetic/mi_rsa_utils.h"


using namespace medical_imaging;
namespace
{
    static const char* S_N = "9C29F9F0057AF656E2C262F0A3110CB4A42027F646EF590F8A6AE927DC08D82990295587C98C7EF0D094A1C6D3D1B77C271C76BB8FCF4C1EF18C32F8AFDE81F0007F06BA2B6EC78E532EE2336804D3E55E75BE355F845CEC7B508731F65C3F7B869AB8E8552B8C5487DB7304F40829D309C171C1FEF41E7AF62B490B4F602B2D76B70A4BBB7DA32A27102DA38995120576D16F86FB41B6BFE0C1B6EEFF88139ECD2D9B00595B383AD065F3A3854AC38D83FB2302F009EC550CF202E0AD047E3C852DBD701CB9B549000D8BB5F0283919BEEC0235628F209562859392727CD24A51A331D6B82CD7B519FE7943C587046C5054AA156B62BAFF0619360A4B19C6F1";
    static const char* S_E = "010001";
    static const char* S_D = "825B9BF2DB70488040EC09515C9DD7E056FD2CBEAD7A10FB230D99B197409EC91B3915D7B2CC200EFFDE82A9099A8FB308A6FF0A14C2F3850517865907DE12D378669104094B81337286B976360789A677528C43DB870F56AC9D8D2D8CBC7695B3C82640731056251DFF7725145C58257D884860AB65B6A7E8243BE6400D441C706F2B4E45B894878E491BE0882980B97A487F66C0E99855E59C5AB15C4F8085852278A49A005232079EA7433164B86FA12504A3ED42DC95BC86DC23600E64A21609DE9BEDFA0B4BA9C79BC4F8F21C60F901C84986BB9F4A6B73EE3097BC7713B2C5A310E37DFF4899CC9467DB9C12963DE75C4B9BC039C418BBB318F2FEE21D";
    static const char* S_P = "E04957B1E9FC84C3719F1B6FF1247C5FF674CDA3334C08AF219D600488D272C218347452665EE8270C251DE14609B8AC24AE56918219550F4E30D1B9BB179E7ABC06D4936940AD04FB7C0003256E50DD517F9A2ECF1B78187E5BA13227CE4FC3F82F60C70D3538F8B012791068D502047F63B59E7BE38448D904D7CA13A9CBAB";
    static const char* S_Q = "B23EC11FB648D8CEB39AAE23293561607975B143EDC4A94A40C1EDC2936B3025C4E82A676D2D27FEF19E18F3BABDBA571D4E98B63575A4A38F07D48A2025422C6655F30813B99A63697E81CC71019991AABC2E3F2C93D92049219878BCD651AB9A7A10832D94CF306C166EEC9F56088D8A307F207DCBBC1806B7DB12A223BBD3";
    static const char* S_DP = "85B9D50AA42B46714D7683226C51C7C263ACE2CAF293DBFDA77A30BCA3636EBEE135AD414FFE3846C7CBFD93CA71936537FDA669DD7B03273C04899746C0DF61E867DE29023168B7B6C6092FD70A7E36671840B2B61377B88AE65127196ABE4E66D4C0CC0DC8F4EDF9F519AFBA017ED175AEDAA3E2D4159465A8A88CB8CA9D3F";
    static const char* S_DQ = "4E9EC57A6261D157A2FAE832541BDA7EBE343E6332FE1A99C8E48125E0F6577F6151F25A3A5ABF98812475E713885A27D0A279536D531DB29305262762B46C72BF14CC24D4E67A05BD63728725954A126957A5A271DC28DA47C78CC43CAEDFC92C5308F383686ED6F1E6173941B2A605205DF1C4F817A43888C611D82F3249C7";
    static const char* S_QP = "BF5515AF98FFB8E18CE134236A84D19CEB0D27C658A0C086AA0172ECB9B095D2F204951069AA24E0681E4953C941473DD86C1C20196BCA65E6A99B50D232ACC512C0B1837FAA28F38D1BCD399303F8DF2406C2BA7DC4817AF08160C8DA44103F86EA8BCD7D1222E6C66D9D063EA1200F55A65DA0F919FB22093219809C023B21";
}

void UT_RSA()
{
    int status = 0;
    mbedtls_rsa_context rsa;

    RSAUtils rsa_tools;

    /*status = rsa_tools.gen_key(rsa);

    if(status != 0)
    {
        std::cout << "Generate key failed!\n";
    }

    rsa_tools.key_to_file(rsa , "D:/temp/rsa_pub.txt" , "D:/temp/rsa_priv.txt");*/

    status = rsa_tools.to_rsa_context(S_N , S_E , S_D , S_P , S_Q , S_DP , S_DQ , S_QP , rsa);
    if(status != 0)
    {
        std::cout << "to rsa context failed!\n";
    }

    std::string pos_x("45.62359");
    std::string diameter("0.2663154");
    std::string nodule_type("ACC");

    unsigned char entrypt_output_pos_x[512];
    unsigned char entrypt_output_diameter[512];
    unsigned char entrypt_output_nodule_type[512];

    status = rsa_tools.entrypt(rsa , pos_x.size() ,(unsigned char*)(pos_x.c_str()) , entrypt_output_pos_x);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    status = rsa_tools.entrypt(rsa , diameter.size() ,(unsigned char*)(diameter.c_str()) , entrypt_output_diameter);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    status = rsa_tools.entrypt(rsa , nodule_type.size() , (unsigned char*)(nodule_type.c_str()) , entrypt_output_nodule_type);
    if(status != 0)
    {
        std::cout << "entrypt failed!\n";
    }

    unsigned char detrypt_output_pos_x[1024];
    unsigned char detrypt_output_diameter[1024];
    unsigned char detrypt_output_nodule_type[1024];

    memset(detrypt_output_pos_x , 0 , sizeof(detrypt_output_pos_x));
    memset(detrypt_output_diameter , 0 , sizeof(detrypt_output_diameter));
    memset(detrypt_output_nodule_type , 0 , sizeof(detrypt_output_nodule_type));


    status = rsa_tools.detrypt(rsa ,  512 , entrypt_output_pos_x , detrypt_output_pos_x);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }
    status = rsa_tools.detrypt(rsa , 512 , entrypt_output_diameter , detrypt_output_diameter);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }
    status = rsa_tools.detrypt(rsa , 512 , entrypt_output_nodule_type , detrypt_output_nodule_type);
    if(status != 0)
    {
        std::cout << "detrypt failed!\n";
    }

    std::string result_pos_x((char*)detrypt_output_pos_x);
    std::string result_diameter((char*)detrypt_output_diameter);
    std::string result_nodule_type((char*)detrypt_output_nodule_type);

    std::cout << result_pos_x << "\n";
    std::cout << result_diameter << "\n";
    std::cout << result_nodule_type << "\n";

}